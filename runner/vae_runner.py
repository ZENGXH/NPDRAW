import os
import sys
import time
import random
import pickle
from datetime import datetime, timedelta 
from PIL import Image
import numpy    as np
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm 
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid

import utils 
from utils import data_helper, model_helper, io_helper, helper 
from utils.model_helper import MODEL_LIST_NO_CANV 
from utils.metric_logger import MetricLogger 
from utils.checker import *
from utils.pytorchtools import EarlyStopping

TRAIN_SUBSET=0
DO_SAMPLE=True

class Runner(object):
    def __init__(self, cfg, metric, local_rank, sample_only=False):
        self.local_rank = local_rank
        self.cfg = cfg
        self.best_modelpath = None
        self.last_msg_train, self.last_msg_eval, self.n_xid_train = '','',0
        self.img_size = data_helper.get_imgsize(cfg.dataset) 
        self.canvas_size = data_helper.get_canvsize(cfg.dataset) 
        self.imgd = data_helper.get_imgd(cfg.dataset)

        assert( self.cfg.cat_vae.canvas_dim > 0)
        self.canvasd = self.cfg.cat_vae.canvas_dim

        logger.debug('img_size: {} | imgd: {} | local_rank: {}', self.img_size, self.imgd, self.local_rank)
        self.metric = MetricLogger() if metric is None else metric 

        if not cfg.distributed:
            self.device = torch.device("cuda") 
        else:
            self.device = torch.device("cuda:%d"%local_rank)
        self.input_dim = data_helper.get_imgsize(cfg.dataset) ** 2
        if not sample_only: self.init_data_loader() 

        # select model 
        self.model = self.build_model()
        self.model.set_metric(self.metric)
        self.model.n_xid_train = self.n_xid_train
        
        if cfg.distributed: 
            self.model = model_helper.DataParallelPassthrough(
                self.model, device_ids=[local_rank], 
                output_device=local_rank, 
                broadcast_buffers=False, find_unused_parameters=True).to(
                torch.device("cuda:%d"%local_rank)) 

        ## special setting for some model: 
        if cfg.use_prior_model:
            self.model.set_bank(self.model.device)
        self.max_epoch = cfg['epochs']
        if not sample_only:
            self.model.num_total_iter = self.max_epoch * len(self.train_loader)
        self.init_epoch = 0
        if not sample_only: self.init_optimizer()  
        self.test_loss_best = 1e7  
        self.best_epoch = 0
        self.model.train() 
        self.metric.log_model(self.model) 
        self.dict_msg_eval, self.dict_msg_train = {}, {}

    def init_data_loader(self):
        cfg = self.cfg 
        SPLIT_TRAINVAL = data_helper.split_val_from_train(cfg.dataset) 
        # select dataset 
        kwargs = {'num_workers': 1, 'pin_memory': False} ## if not cfg['no_cuda'] else {}
        train_set = helper.build_data_set(cfg.dataset, 1) # - cfg.inverse_train)
        data_helper.label2imgid(train_set)
        # TODO: for debuging 
        num_train = len(train_set)
        self.test_set_label_offset = num_train 
        if TRAIN_SUBSET:
            selected = list(range(min(len(train_set), cfg.batch_size*10+1000)))
            train_set = torch.utils.data.Subset(train_set, selected)

        if SPLIT_TRAINVAL:
            selected = list(range(0, len(train_set)-1000)) 
            # select last 1k sample as validation set 
            ntest = 1000 if cfg.test_size == -1 else cfg.test_size 
            val_set = torch.utils.data.Subset(train_set, 
                    list(range(len(train_set)-ntest, len(train_set))))
            # the sample index of val-loader is offset by zero in 
            # the full prior_gt; since itself is the subset of 
            # the train-set 
            self.val_set_label_offset = 0 # used to do gt indexing   
            # update the training set | train_set is the [0,-1k] samples  
            train_set = torch.utils.data.Subset(train_set, selected)
            # create full test set 
            test_set = helper.build_data_set(cfg.dataset, istrain=0) #cfg.inverse_train) 
        else:
            td = data_helper.get_test_data(cfg.dataset)
            test_set = helper.build_data_set(td, istrain=0) #cfg.inverse_train) 
            val_set  = helper.build_data_set(cfg.dataset, istrain=0) #cfg.inverse_train) 
            logger.info('[Non SPLIT_TRAINVAL] num of val={}', len(val_set))
            self.val_set_label_offset = self.test_set_label_offset
        self.n_xid_train = len(train_set)

        if cfg.distributed: 
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
            self.train_loader = torch.utils.data.DataLoader(train_set, 
                sampler=self.train_sampler,
                batch_size=cfg['batch_size'], **kwargs)
        else:
            self.train_loader = torch.utils.data.DataLoader(train_set, 
                batch_size=cfg['batch_size'], shuffle=True, **kwargs)
        
        # 0: eval on test; 1: eval on train  
        ## self.test_set_label_offset = 0
        data_helper.label2imgid(test_set)
        if not SPLIT_TRAINVAL: # otherwise its called above 
            data_helper.label2imgid(val_set)
            if len(val_set) > 1000:
                val_set = torch.utils.data.Subset(val_set, list(range(0,1000))) 

        logger.info('[NUM of image] in train:val:test={}:{}:{} | val_set offset={}', 
            len(train_set), len(val_set), len(test_set), self.val_set_label_offset)
        if cfg.distributed: 
            self.full_test_loader = torch.utils.data.DataLoader(test_set, 
                sampler=torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False),
                batch_size=cfg['test_batch_size'], **kwargs)
        else:
            self.full_test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=cfg['test_batch_size'], shuffle=False, **kwargs)
        self.num_sample = 50 

        if SPLIT_TRAINVAL:
            self.num_sample = 50 
            logger.info('SPLIT_TRAINVAL == 1, '
                'use last 1k sample in training set as validation | Nsample '
                'to estimate val NLL = {}', self.num_sample)
            pass
        elif cfg.test_size > 0:
            selected = list(range(len(val_set)))
            selected = selected[:cfg.test_size]
            val_set = torch.utils.data.Subset(val_set, selected)
            self.num_sample = 5 
        if self.img_size == 256: 
            self.num_sample = 2
        if cfg.distributed: 
            self.val_loader = torch.utils.data.DataLoader(
                val_set, sampler=torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False),
                batch_size=cfg['test_batch_size'], **kwargs)
        else:
            self.val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=cfg['test_batch_size'], shuffle=False, **kwargs)

        logger.info('build data with shape: {}; batch size {}'.format(
            train_set[0][0].shape, cfg['batch_size']))


    def init_optimizer(self): 
        # build optimizer  
        cfg = self.cfg 
        self.optimizer = self.model.get_optim(cfg.lr) 
    def build_model(self):
        cfg = self.cfg 
        #from model.cat_vae_fixed_loc import VAE as CatVaeFixedLoc  
        if cfg.model_name in ['vae', 'cvae', 'cvae2']:
            from model.vae import VAE as Model
        elif cfg.model_name == 'cat_vloc_at': # use bank, vary loc 
            from model.vary_loc_at import CatVaryLocAT as Model 
        else:
            raise ValueError('Not support %s'%cfg.model_name)
        built_model = Model(cfg)
        built_model.to(self.device)
        return built_model 

    def train_epochs(self):
        logger.info('start training from E{} to {}', 
            self.init_epoch, self.max_epoch)
        EVAL = os.getenv('EVAL', None)
        cmt = self.cfg.cmt 
        if EVAL: 
            cmt += ' [EVAL] %s'%EVAL
        outdir = os.path.join(self.cfg.exp_dir, self.cfg.exp_name) 

        pre_msg = f'\n[CMT]: {cmt} \n{outdir} '
        slurm_id = os.getenv('SLURM_JOB_ID', None)
        slurm_name = os.getenv('SLURM_JOB_NAME', '' )
        slurm_node = os.getenv('SLURM_JOB_NODELIST', '')
        if len(self.metric.comet_url) > 1:
            pre_msg += f'\n[url]: {self.metric.comet_url}'
        else:
            pre_msg += f'\n {self.cfg.exp_key}'

        if slurm_id is not None:
            pre_msg += ' |[jid] %s, %s, %s'%(slurm_id, slurm_name, slurm_node)

        if not self.local_rank: 
            logger.info(pre_msg)

        teloss = self.test_loss_best 
        t0 = time.time() 
        t1 = time.time() 
        nwrite = 0
        init_epoch = self.init_epoch
        epoch = init_epoch-1
        for epoch in range(init_epoch, self.max_epoch): 
            if self.cfg.distributed: 
                self.train_sampler.set_epoch(epoch) 
            tic = time.time() 
            self.metric.start_epoch(epoch)
            #if epoch == 0:
            #    self.sample(epoch)
            #    self.metric.eval() 
            #    teloss = self.test(epoch, self.num_sample)
            self.metric.train() 
            self.model.train() 
            # -- train epoch -- 
            trloss = self.train(epoch)
            epoT = (time.time() - tic)/60 

            # -- fill log msg -- 
            msg = pre_msg
            if self.dict_msg_eval.get(self.best_epoch):
                msg += '\n[best]: ' + self.dict_msg_eval[self.best_epoch]
            self.last_msg_train = '{} | eT:{:.2f}m'.format(self.metric.msg('train', len(self.train_loader)), epoT)
            if self.local_rank == 0:
                logger.info('\n{} | \n{}', self.last_msg_train, msg)

            # -- evaluation -- 
            if (1+epoch) % self.cfg.test_interval == 0:
                self.metric.eval() 
                teloss = self.test(epoch, self.num_sample)
            if self.local_rank: continue 
            if DO_SAMPLE: 
                self.sample(epoch)

            # -------------
            # snapshot model 
            if teloss < self.test_loss_best:
                self.test_loss_best = teloss 
                modelpath = self.save_model('ckpt_best_eval.pth', epoch=epoch) 
                self.best_modelpath = modelpath 
                logger.info('>'*10 + '[best eval %.1f at epo %d]'%(
                    teloss, epoch))
                self.best_epoch = epoch 
                self.metric.write('best_nll', teloss)
                if epoch in self.dict_msg_eval:
                    self.metric.log_html(self.dict_msg_eval[epoch])
                    self.metric.write('best', self.dict_msg_eval[epoch])
            else: 
                logger.info(f'cur E{epoch} {teloss:.3f} | Best E={self.best_epoch}; loss={self.test_loss_best:.3f}: ') 

            if epoch % 50 == 0 :
                slurm_dir=f"/checkpoint/{os.getenv('USER')}/{os.getenv('SLURM_JOB_ID', None)}"
                savedp = self.save_model('E%05d.pth'%epoch, epoch=epoch) #, expdir=slurm_dir)
            elif (time.time() - t0) / 60 > 10: # more than 30 mins 
                t0 = time.time()
                logger.info('*'*10 + 'snapshot model' + '*'*10)
                self.save_model('snapshot.pth', epoch=epoch) 
            self.init_epoch += 1
            # --- end of one epoch --- 
        if not self.local_rank: 
            self.save_model('ckpt_epo%d.pth'%epoch, epoch=epoch)
            self.save_model('snapshot.pth', epoch=epoch) 
        logger.info('done training')

    def load_model(self, ckpt):
        loaded_model_dict = {}
        for k,v in ckpt['model'].items():
            if 'module' in k: 
                k = k.replace('module.', '')
            loaded_model_dict[k] = v
        self.model.load_state_dict(loaded_model_dict) 
        self.init_epoch = ckpt['epo'] + 1 
        self.test_loss_best = ckpt['test_loss'] if 'test_loss' in ckpt else 0

    def save_model(self, modelpath, epoch=0, expdir=None):
        modelpath = os.path.join(expdir, modelpath) if expdir is not None else \
            os.path.join(self.cfg.exp_dir, self.cfg.exp_name, modelpath) 
        if not os.path.exists(os.path.dirname(modelpath)):
            os.makedirs(os.path.dirname(modelpath))
        snapshot = {'model': self.model.state_dict(), 
            'optim': self.optimizer.state_dict(), 'cfg':self.cfg, 
            'test_loss': self.test_loss_best, 'epo':epoch, 'best_epo':self.best_epoch} 
        torch.save(snapshot, modelpath)
        logger.info('[save] model as %s'%modelpath)
        return modelpath

    @torch.no_grad()
    def sample(self, epoch):
        tic = time.time()
        self.model.eval()
        if self.cfg.model_name in MODEL_LIST_NO_CANV: 
            hid=torch.randn(64, *self.model.latent_shape) 
            sample = self.model.sample(hid.to(self.model.device)) 
        else: 
            sample = self.model.sample() 
            hid = None
        if type(sample) is tuple: 
            sample, hid = sample
        B = 64
        nrow = B 
        sample = sample.cpu()
        canvas = hid 
        if canvas is not None and self.cfg.model_name not in MODEL_LIST_NO_CANV: 
            vis_shape = [self.img_size,self.img_size]
            comparison = [sample.view(B,self.imgd,*vis_shape)]
            empty_line = sample.new_zeros(nrow, self.imgd, *vis_shape) + 0.5 
            if type(canvas) is list: 
                canvas = [c.cpu() for c in canvas]
                for c in canvas: 
                    comparison.append(empty_line)
                    comparison.append(c.view(B,self.imgd,*vis_shape))
            else: 
                canvas = canvas.cpu() 
                if canvas.shape[-1] != self.img_size:
                    canvas = F.interpolate(canvas, self.img_size, mode='bilinear')
                sp_img = [B,self.imgd,self.img_size,self.img_size]
                sp_canv = [B,self.canvasd,self.img_size,self.img_size]
                comparison = [sample.view(*sp_img), empty_line,
                    canvas.view(*sp_canv).expand(*sp_img)] 
                # along B 
            comparison = torch.cat(comparison) 
            fig = make_grid(comparison.cpu(), pad_value=0, nrow=nrow, normalize=True, scale_each=True)
        else:
            fig = make_grid(sample.view(B,self.imgd,self.img_size,self.img_size), pad_value=0, nrow=nrow) 

        if epoch == -1:
            tag = 'sample_%d'%(self.init_epoch + epoch)
        else:
            tag = 'sample'
        filename = '%s/%s/'%(self.cfg.exp_dir, self.cfg.exp_name) + 'vae_%s_'%tag + str(epoch) + '.png'
        if not self.metric.log_made_grid(tag, fig, epoch) or epoch == -1:
            fig = save_image(fig, filename)    
            logger.info('save img at {}', filename)

    @torch.no_grad()  
    def test(self, epoch, num_sample=50):
        self.model.eval()
        if not self.local_rank: self.vis_recont(epoch) 
        nll = self.compute_elbo_nsample(epoch, write_out=False, num_sample=num_sample) 
        return nll

    @torch.no_grad()
    def compute_elbo_nsample(self, epoch, write_out=True, num_sample=50):
        """ used for evaluation, will call model's test_loss function """
        self.model.eval()
        self.metric.start_epoch(epoch)
        tic = time.time() 
        output_dir = os.path.join(self.cfg.exp_dir, self.cfg.exp_name) 
        test_nll = []
        cnt = 0
        if write_out: # final eval, load full test set 
            self.metric.test() 
            tname = 'test'
            test_loader = self.full_test_loader 
            label_offset = self.test_set_label_offset
        else: # train-eval stage, load val set 
            self.metric.eval() 
            tname = 'eval'
            test_loader = self.val_loader
            label_offset = self.val_set_label_offset
        # Test with original estimator 
        for i, (data, labels) in enumerate(test_loader):
            Bs = len(data)
            self.model.set_xid( labels.to(self.model.device) + label_offset)  
            cnt += Bs
            data = data.to(self.model.device).float() #.view(1,B,784)
            output, loss_dict = self.model.test_loss(data, num_sample=num_sample)
            test_nll.append(loss_dict['NLL'].item())
            self.metric.update(**loss_dict) 
            if (i+1) % 2000 == 0 and self.local_rank == 0:
                logger.info('ns: {} {}', num_sample, self.metric.msg('test', i+1)) 
        msg = '{} | cnt={}'.format(
            self.metric.msg(tname, len(test_loader)), cnt) 
        self.dict_msg_eval[epoch] = msg 
        self.last_msg_eval = msg 
        if not write_out: 
            if not self.local_rank: logger.info(' ~~ ' + 'eval: {} ', msg) 
        else:
            pre_msg = '\n' + '--'*10 + '\n' + '[cmt]: ' + self.cfg.cmt + '\n' 
            if self.last_msg_train != '':
                pre_msg += self.last_msg_train + '\n' 
            if self.dict_msg_eval.get(self.best_epoch) and epoch != self.best_epoch:
                pre_msg += self.dict_msg_eval.get(self.best_epoch) + '\n' 
            msg = pre_msg + msg + '\n' + f'{self.metric.comet_url}'
            msg += '\n %s \n %s'%(output_dir, '--'*10) 
            if not self.local_rank: logger.info(msg)
        return np.mean(test_nll)

    @torch.no_grad()
    def vis_recont(self, epoch):
        output_dir = os.path.join(self.cfg.exp_dir, self.cfg.exp_name) 
        tag = 'recont'
        filename = output_dir + '/eval_%s_%d.png'%(tag, epoch)
        img_size = self.img_size 
        NVIS=8 
        data_list, label_list = [], []
        for i, (data, labels) in enumerate(self.full_test_loader):
            data = data.to(self.device).float() #.view(1,B,784)
            labels = labels.to(self.device) + self.test_set_label_offset  
            data_list.append(data)
            label_list.append(labels) 
            if len(data_list) * data.shape[0] > NVIS: 
                break 
        data, labels = torch.cat(data_list), torch.cat(label_list)
        self.model.set_xid(labels[:NVIS])
        data = data[:NVIS]
        img = self.model.vis(data)
        img = img.view(-1,self.imgd,img_size,img_size) 
        fig_grid = make_grid(
            torch.cat([img, data.view(-1,self.imgd,img_size,img_size)]), 
            nrow=NVIS, normalize=True, scale_each=True,
            ) 
        if not self.metric.log_made_grid(tag, fig_grid, epoch): 
            fig = save_image(fig_grid, filename)    
            logger.info('save img at {}', filename)
      
    @torch.no_grad() 
    def sample_10k(self, epoch): 
        self.model.eval()
        out = []
        out_hid = []
        N = 64 
        logger.info('start sampling, N={}', N)
        self.model.sample_10k = False # temp solution to turn off z&e controller 
        for k in tqdm(range(10000 // N + 1)):
            # 10000 / 64 
            if self.cfg.model_name in MODEL_LIST_NO_CANV: 
                hid=torch.randn(64, *self.model.latent_shape) 
                sample = self.model.sample(hid.to(self.device)) 
            else:
                sample = self.model.sample() 
                hid = None

            if type(sample) is tuple: 
                sample, hid = sample

            B = sample.shape[0]
            sample = sample.cpu().view(B, self.imgd, self.img_size, self.img_size)
            out.append(sample) 
            if k < 1:
                filename = '%s/%s/'%(self.cfg.exp_dir, self.cfg.exp_name) + \
                        'sample10k_' + str(epoch) + '-%d.png'%k
                if hid is not None and hid.shape[-1] == self.img_size: 
                    hid = hid.cpu()
                    if hid.shape[1] < self.imgd:
                        hid = hid.expand(-1,self.imgd,-1,-1)
                    if self.canvas_size < self.img_size:
                        hid = F.interpolate(hid, self.img_size, mode='bilinear')
                    sample = torch.cat([sample, hid])
                fig = save_image(sample, filename, normalize=True, scale_each=True, nrow=8)    
        out_pt = torch.cat(out) 
        logger.info('get output: {}', out_pt.shape) 
        assert(out_pt.shape[0] > 10000), 'get output less than 10k sample'

        out = out_pt.numpy() 
        filename = '%s/%s/'%(self.cfg.exp_dir, 
                self.cfg.exp_name) + '10k_sample' + str(epoch) + '.npy'
        np.save(filename, out)
        logger.info('save at %s'%filename)
        return filename 
