import sys 
import numpy as np
from loguru import logger
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.distributions.categorical import Categorical
from model.srvae_modules.distributions import dmol_loss, sample_from_dmol 
from utils import data_helper 
from utils.checker import *

class Base(nn.Module):
    def __init__(self):
        super().__init__() 
        self.xid = None
        self.sample_10k = False
        self.metric = None 
        self.temp_gt_q = 1.0 # temperature of giving gt for Q 
        self.output_shape = None
        self.has_bn_loss_enc = 0 # default is zero 
        self.has_bn_loss_dec = 0
        self.need_bn_loss = 0
        self.need_sn_loss = 0
        self.has_init_bn = 0
    
    def set_xid(self, xid):
        self.xid = xid 
    
    @property 
    def device(self):
        param = next(self.parameters())
        return param.device  

    def set_metric(self, metric):
        self.metric = metric 

    def get_patch_bank(self):
        # return patch bank in shape K,H,W,D 
        if 'patch_bank' not in self.__dict__:
            raise ValueError 
        pb = self.patch_bank # expected to be 1,nH,nW,K,pH,pW,canvD 
        CHECKSIZE(pb, (1,-1,-1,-1,self.patch_size,self.patch_size,
            self.canvasd))
        return pb[0,0,0] # return K,H,W,D

    def init_data_param(self):
        self.patch_size = self.cfg.ww
        self.pred_locvar = 0
        self.out_dist = data_helper.get_out_dist(self.cfg.dataset)
        self.canvas_size = data_helper.get_canvsize(self.cfg.dataset) 
        self.n_class_loc = (self.canvas_size // self.cfg.vary_loc_vae.loc_stride)**2
        self.imgd = data_helper.get_imgd(self.cfg.dataset)
        self.img_size = data_helper.get_imgsize(self.cfg.dataset)
        self.pix_class = data_helper.get_pixel_nclass(self.cfg.dataset)
        self.out_dist_npara = data_helper.get_out_dist_npara(self.out_dist, self.pix_class)
        self.input_dim = (self.img_size**2)*self.imgd 
        self.padl, self.padr = data_helper.get_floc_pad(self.cfg.dataset, self.cfg.ww)
        self.out_imgshape = (self.imgd, self.img_size, self.img_size)

        self.canvasd = data_helper.get_canvasd(self.cfg)
        #if self.cfg.spe_norm:
        #    self.need_sn_loss = 1
        #if self.cfg.cat_vae.canvas_dim > 0:
        #    self.canvasd = self.cfg.cat_vae.canvas_dim
        #else:
        #    self.canvasd = self.imgd 
        # logger.info('*'*10)
        # logger.info(f'[out dist]: {self.out_dist}, npara={self.out_dist_npara} '
        #        f'| imgd: {self.imgd} | ')
        logger.info(f'imgd={self.imgd}, img_size={self.img_size}, '
                f'canv_size={self.canvas_size}, canv_d={self.canvasd} '
                f'outdist={self.out_dist}, require #para={self.out_dist_npara}')
        if self.out_dist == 'gaussian_as2svae':
            self.log_gamma = torch.nn.Parameter(torch.ones(1).log(), 
                requires_grad=True) 
            logger.info('INIT log_gamma!')

    def out2sampled(self, out):
        '''
        if cat: out [B,D,H,W,K]
        if logistic: [B,D,H,W,2] 
        if bernoulli: B,H,W
        if mixture_logistic: B,(imgd*3)*nmix + nmix ; for each mixture: 1 for prob, 3(mean.var,gamma)x3(r,g,b)
                [B,100,H,W]
        '''
        B = out.shape[0]
        #if self.cfg.cat_vae.fixed_loc_dec_version == 'identity':
        #    out = out # no processing, the output == canvas 
        if self.out_dist == 'cat':
            out = out.argmax(-1)/255.0 
        elif self.out_dist == 'logistic':
            out = out.view(-1,2)[:,0]
            out = out.clamp(min=0., max=1.) 
        elif self.out_dist in ['gaussian_as2svae', 'gaussian', 'bernoulli', 'l2']:
            out = out 
        elif self.out_dist in ['mixture_logistic']:
            out = out.view(B, -1, self.img_size, self.img_size)
            out = sample_from_dmol(out, nc=self.imgd, random_sample=False)
        else: #if self.out_dist == 'mixlogistic':
            logger.info('out dist {}', self.out_dist)
            raise NotImplementedError 

        out = out.view(-1,self.imgd,self.img_size,self.img_size)
        return out 

    @torch.no_grad()
    def cls(self, x):
        raise NotImplementedError 

    @torch.no_grad()
    def test_loss(self, x, num_sample):
        raise NotImplementedError 

    @torch.no_grad() 
    def vis(self, x):
        raise NotImplementedError 

    @torch.no_grad()
    def sample(self, sample_cfg={}):
        raise NotImplementedError 

    @torch.no_grad()
    def forward(self, x):
        # will be called as: 
        # recon_batch, loss_dict = self.model(data)
        raise NotImplementedError 

    def set_epoch(self, epo): 
        self.epoch = epo 

    def get_optim(self, lr):
        assert(not self.cfg.optim.diff_lr), 'not support'
        if self.cfg.optim.name != 'adam':
            raise ValueError(self.cfg.optim.name)
        logger.info('set optimizer: with lr: {}', lr)
        param_list = []
        for name, param in self.named_parameters(): 
            if param.requires_grad:
                logger.debug('> {}', name)
                param_list.append(param)
        return torch.optim.Adam(param_list, lr)

    def setup_temp(self):
        cfg = self.cfg 
        self.temp_min = cfg.temp_min
        # self.ANNEAL_RATE = 0.0001
        self.ANNEAL_RATE = cfg.temp_anneal_rate 
        # set temp as parameter such that it can be loaded during eval 
        self.temp_init = torch.nn.Parameter(torch.zeros(1) + self.cfg.temp_init, requires_grad=False)
        self.temp      = torch.nn.Parameter(torch.zeros(1) + self.cfg.temp_init, requires_grad=False)
        logger.debug('| temp init {} | ', self.temp_init.item())

    def anneal_tgq(self, i):
        # temp_gt_q 
        self.temp_gt_q = self.temp_gt_q * np.exp(-self.cfg.vary_loc_vae.te_sel_gt_temp * i) 
        return self.temp_gt_q 

    def anneal_per_epoch(self, i):
        self.temp.data = torch.clamp(torch.zeros_like(self.temp.data) + \
                self.cfg.temp_init * np.exp(-self.ANNEAL_RATE*i), 
                min=self.temp_min)
        if 'tau0' in self.__dict__:
            self.tau0.data = torch.clamp(torch.zeros_like(self.temp.data) + \
                self.cfg.temp_init * np.exp(- self.ANNEAL_RATE * i), 
                min=self.temp_min)
        return self.temp.data 

    def compute_recont_loss(self, recon_x, x, B, compute_bpd=True):
        ''' 
        x: B,input_dim, range in [0,1]
        recon_x: B,input_dim, .., npara 
        '''
        out_dist = self.out_dist 
        input_dim = self.input_dim 
        if self.out_dist in ['mixture_logistic']: # B,100,H,W
            x = x.view(B,self.imgd,self.img_size,self.img_size)
            recon_x = recon_x.view(B,-1,self.img_size,self.img_size)
            BCE = - dmol_loss(x, recon_x, nc=self.imgd)  
        elif out_dist == 'bernoulli':
            recon_x = recon_x.view(B, input_dim)
            x = x.view(B, input_dim)
            BCE = F.binary_cross_entropy(
                    recon_x, x, reduction='none').view(B,input_dim).sum(-1) 
        elif out_dist == 'cat':
            x = x * 255 
            BCE = F.cross_entropy(recon_x.view(B*input_dim,-1), x.view(-1).long(), reduction='none').view(B,-1).sum(-1)
        elif out_dist == 'logistic': 
            # expect: recont_x: B,C,H,W,2
            # original img: 
            x = x.view(B*input_dim) # * 255 , range [0,1]
            
            xmax, xmin = 1.0, 0.0 
            assert(x.max() <= xmax and x.min() >= xmin), f'get x range: {x.min()}, {x.max()}'
            rescaled = 256.0
            
            thres_max = (255 - 1) / rescaled
            thres_min = 1 / rescaled   

            recon_x = recon_x.view(B*input_dim, 2)
            x_mu = recon_x[:,0]

            x_s = (recon_x[:,1]).clamp(max=80).exp().clamp(min=1e-7) # minimum value 
            # invx_s = (1 / x_s).clamp(min=1e-7)
            # 1 / x_s = 1 / exp(log_scale) = exp(-log_scale)? 
            CDF_max = torch.sigmoid((x+0.5/rescaled-x_mu)/x_s)  
            CDF_min = torch.sigmoid((x-0.5/rescaled-x_mu)/x_s)
            #CDF_max = torch.sigmoid((x+1.0/rescaled-x_mu)/x_s)  
            #CDF_min = torch.sigmoid((x-x_mu)/x_s)

            right_edge_mask = (x > thres_max).float() # x = 255, P(254.5 < x < 255.5) become: P(254.5 < x) 
            CDF_max = CDF_max * (1-right_edge_mask) + right_edge_mask # for those x > 254, set CDF_max = 1.0  
            left_edge_mask = (x < thres_min).float() # x = 0, P(-0.5<x<0.5) become P(x<0.5)
            CDF_min = CDF_min * (1-left_edge_mask) # for those x <1, set the CDF_min as 0 
            BCE = - ((CDF_max-CDF_min).clamp(min=1e-7).view(B,input_dim)).log().sum(-1)
            # BCE = - ((CDF_max-CDF_min).view(B,input_dim)).clamp(min=1e-12).log().sum(-1)
        elif out_dist == 'gaussian': # expect: B,C,H,W,2
            x = x.view(B*input_dim) # * 255 , range [0,1]
            recon_x = recon_x.view(B*input_dim, 2) 
            # expect: B,D,H,W,2
            x_mu = recon_x[:,0]
            x_s = recon_x[:,1].exp() 
            dist = Normal(x_mu, x_s) 
            BCE = - dist.log_prob(x).view(B,input_dim).sum(-1) 
            
            if compute_bpd:    
                CDF_max = dist.cdf(x + 0.5/256)
                CDF_min = dist.cdf(x - 0.5/256) 
                BCE = - ((CDF_max-CDF_min).clamp(min=1e-7).view(B,input_dim)).log().sum(-1)
            #CDF_max = dist.cdf(x + 0.5/256)
            #CDF_min = dist.cdf(x - 0.5/256) 
            #BCE = - ((CDF_max-CDF_min).clamp(min=1e-7).view(B,input_dim)).log().sum(-1)
        elif out_dist == 'gaussian_as2svae': 
            log_gamma = self.log_gamma
            x = x.view(B*input_dim)
            recon_x = recon_x.view(B*input_dim, 1)
            x_mu = recon_x[:,0]
            assert(log_gamma is not None)
            x_s = log_gamma.exp().view(1).expand(B*input_dim)
            dist = Normal(x_mu, x_s) 
            BCE = - dist.log_prob(x).view(B,input_dim).sum(-1) 
            #+ 0.5/256 * torch.zeros_like(x).uniform_(-1,1))
            if compute_bpd:    
                CDF_max = dist.cdf(x + 0.5/256)
                CDF_min = dist.cdf(x - 0.5/256) 
                BCE = - ((CDF_max-CDF_min).clamp(min=1e-7).view(B,input_dim)).log().sum(-1)
        elif out_dist == 'l2':
            x = x.view(B*input_dim) # * 255 , range [0,1]
            recon_x = recon_x.view(B*input_dim)
            BCE = F.mse_loss(x, recon_x, reduction='none').view(B, input_dim).sum(-1) 
            ## ((x - recon_x)**2).view(B, input_dim).sum(-1)
        else:
            logger.info('get out_dist={}', out_dist)
            raise NotImplementedError
        return BCE 

    def kl_coeff(self, step, num_total_iter):
        total_step = 0.3 * num_total_iter ## total_step 
        constant_step = 0.0001 * num_total_iter
        min_kl_coeff = 0.0001
        self.kl_coeff_v = max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)
