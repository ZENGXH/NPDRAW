''' usage: 
    python ./train_vae.py --comet 0 model_name cat_vae epochs 20 tag 'test' lr 1e-3 batch_size 200 dataset 'stoch_mnist' cat_vae.estimator 'reinforce_bl' 
or 
'''
from loguru import logger
import numpy as np
import torch 
from torch import nn 
import torch.nn.functional as F 
import os
import torchvision 
from utils import model_helper 
from .vae_runner import Runner as BaseRunner 

def aggregate_metric(metric, output): 
    # collect print only metric 
    for k in output.keys():
        flag = ['_print', '_time'] 
        for f in flag:
            if f not in k: continue 
            ks = k.split(f)[0]
            if f == '_time': 
                metric.update(**{ks: output[k]}) # convert dict to kwargs 
            else:
                metric.update(**{ks: output[k].mean(0)}) # convert dict to kwargs 


def aggregate_loss_gs(args, output, metric, model, names=None): 
    loss = 0 
    if names is not None: 
        assert(names in output), f'{names} not in {list(output.keys())}'
        metric.update(BCE=output[names].mean(0)) 
        loss += output[names].mean() # shape: B,1 
    if 'loc_loss' in output: 
        metric.update(loc_loss=output['loc_loss'].mean(0)) 
        loss += output['loc_loss'].mean() * model.cfg.vary_loc_vae.latent_loc_weight 
    if 'sel_loss' in output: 
        metric.update(sel_loss=output['sel_loss'].mean(0)) 
        loss += output['sel_loss'].mean() * model.cfg.vary_loc_vae.latent_sel_weight 
    if 'stp_loss' in output: 
        metric.update(stp_loss=output['stp_loss'].mean(0)) 
        loss += output['stp_loss'].mean() * model.cfg.vary_loc_vae.latent_sel_weight 
    return loss

def step_iter(model, x, optim, args, metric, epoch, cfg):  
    B = x.shape[0]
    optim.zero_grad()
    assert(args.estimator in ['gs', 'stgs'])
    output, _ = model.compute_elbo(x)  
    neg_elbo = output['ELBO']
    metric.update(BCE=output['BCE'].sum()/B, KL=output['KLD'].sum()/B, ELBO=neg_elbo.mean(0))
    # loss will be the ELBO entry 
    loss = torch.mean(neg_elbo) 
    loss += aggregate_loss_gs(args, output, metric, model) 
    aggregate_metric(metric, output) 
    loss.backward()
    if cfg.distributed: 
        model_helper.average_gradients(model)
    optim.step()
    return loss.item(), 0

class DiscreteRunner(BaseRunner):
    def init_optimizer(self): 
        logger.info('Init Optimizer!')
        self.optimizer = self.model.get_optim(self.cfg.lr) 

    def __init__(self, cfg, metric, local_rank, sample_only=False):
        super().__init__(cfg, metric, local_rank, sample_only) 
        args = cfg.cat_vae
        if args.estimator in ('gs', 'stgs'):
            assert args.n_samples == 1
        n_samp = args.n_samples
        
    def train(self, epoch): 
        logger.debug('<'*10)
        # self.model.train() 
        total_iter = len(self.train_loader)
        cost_list = []
        if 'gs' in self.cfg.cat_vae.estimator: 
            if not hasattr(self.model, 'anneal_per_epoch'):
                raise NotImplementedError 
            new_t = self.model.anneal_per_epoch(epoch)
            self.metric.update(temp=new_t)  
        if self.cfg.vary_loc_vae.te_sel_gt == 4:
            tgq = self.model.anneal_tgq(epoch) 
            self.metric.update(temp_gt_q=tgq)
        device = self.model.device 
        logger.info('len of loader={}, rank: {}', len(self.train_loader), 
            self.local_rank)
        total_iter = self.max_epoch * len(self.train_loader)
        for batch_idx, (data, xid) in enumerate(self.train_loader):
            self.metric.update(lr=self.optimizer.param_groups[0]['lr']) 
            B = data.shape[0]
            assert(data.max()<=1), f'max={data.max()}, shape={data.shape}'
            train_xs = data.to(device).view(B,-1).float()
            xid = xid.to(device)
            self.model.set_xid(xid)

            cost, tr_ap = step_iter(self.model, train_xs, self.optimizer, 
                        self.cfg.cat_vae, self.metric, epoch, self.cfg) 

            self.metric.update(cost=cost)
            cost_list.append(cost)
            if self.local_rank: continue # skip the vis part 
            if (batch_idx) % self.cfg.log_interval == 0:
                logger.info('{:04d}/{}: {}', batch_idx, total_iter, self.metric.msg()) 
            if batch_idx == 0:
                self.model.vis_prior_output = True
                NVIS = min(16,B)
                img_size = self.img_size 
                self.model.set_xid(xid[:NVIS])
                img = self.model.vis(train_xs[:NVIS]).view(-1,self.imgd,img_size,img_size)         
                directory = '%s/%s/'%(self.cfg.exp_dir, self.cfg.exp_name) 
                vis_out = torchvision.utils.make_grid(torch.cat(
                     [img, train_xs[:NVIS].view(-1,self.imgd,img_size,img_size)]), nrow=NVIS,
                     pad_value=0.5, normalize=True, scale_each=True) 
                if self.metric.log_made_grid('train', vis_out, epoch) is None:
                    logger.info('save image at {}', directory)
                    torchvision.utils.save_image(
                        vis_out, directory+'/train_recont_%d-%d.png'%(epoch, batch_idx),
                        normalize=True) 

                self.model.vis_prior_output = False
        return np.mean(cost_list)
