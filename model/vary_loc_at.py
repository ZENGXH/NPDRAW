import os
import numpy as np
from loguru import logger
import time 
import math 
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
import utils
from utils.checker import *
from functools import partial
from utils import ph_helper
from utils.ph_helper import get_gt_sel_loc, load_prior_with_patch_bank 
from model.transform import CanvasPlotter 
from model.transform import zcls_BHWK_to_psel_BHW_phw, coords_ij_to_distmap, transform_ij
from model.vary_loc_at_modules import build_model 
from model.modules import gumbel_softmax
from model.base import Base 

import torch.distributions as D 
def logmeanexp(inputs, dim=1):
    if inputs.size(dim) == 1:
        return inputs
    else:
        input_max = inputs.max(dim, keepdim=True)[0]
        return (inputs - input_max).exp().mean(dim).log() + input_max

def draw_sample(logits_y, estimator=None, temp=None): 
    ''' # do categorical sampling
    Ban this function: since we require y_logits_2_flat sampled recurssively?? 
        perform sampling, given logits_y, 
        sampled the 1-of-k hot vector 
    Args: 
        logits_y: B,...,K 
        estimator (str): if gs, return soft vec 
    '''
    # return sampled y: [B,K]
    CHECK2D(logits_y)
    B,K = logits_y.shape 
    if estimator == 'gs':
        y_sample = gumbel_softmax(logits=logits_y, temperature=temp, hard=False) 
        y_flat = y_sample.view(-1)
    elif estimator == 'stgs':
        y_sample = gumbel_softmax(logits=logits_y, temperature=temp, hard=True) 
        y_flat = y_sample.view(-1)
    else:
        raise NotImplementedError
    return y_flat.view(B,K)

def draw_ber(probs, temp=None): 
    return torch.bernoulli(probs).float() 

class CatVaryLocAT(Base):
    model_name = 'cat_vloc_at'
    auto_regressive_posterior = False 
    discrete_loc = True 
    def __init__(self, cfg):
        super().__init__()
        self.loc_dist = 'cat' 
        self.n_class_loc = None
        self.prior_valid_step = None
        self.prior_loc_var = None
        self.z_dim = 0 
        self.cfg = cfg 
        self.add_stop = cfg.cat_vae.add_stop 
        self.setup_temp() 
        self.init_data_param() 
        # parse size:
        self.img_np = (self.img_size**2) * self.imgd 
        self.canv_d = self.canvasd 
        self.nloci = cfg.vary_loc_vae.nloc
        self.nlocj = 1 
        self.latent_dim = self.nloci*self.nlocj 
        self.categorical_dim = self.cfg.K
        self.N = self.latent_dim 
        self.K = self.categorical_dim
        logger.info('categorical_dim={}, nloc={}', self.categorical_dim, self.nloci*self.nlocj)
        # build model: 
        self.try_load_float_matrix() 

        self.build_model() 
        self.xid = None
        self.info_dict = {'categorical_dim': self.K, 'nloci': self.nloci,
            'nlocj':self.nlocj, 'ph': self.cfg.wh, 'pw':self.cfg.ww, 
            'canvas_size':self.canvas_size, 'img_size':self.img_size, 
            'latent_dim': self.latent_dim, 'imgd':self.imgd, 
            'canvasd': self.canvasd}
        self.pred_locvar = cfg.vary_loc_vae.pred_locvar 
        if self.pred_locvar:
            assert(not cfg.vary_loc_vae.latent_loc_sample), 'not support'
        assert(self.float_matrix is not None and self.n_class_loc is not None) 

    def build_model(self): 
        ''' will be called in Base.class, need to set encoder and self.decoder '''
        self.encoder, self.decoder = build_model(self.cfg, self.n_class_loc)

    def try_load_float_matrix(self):
        float_matrix_in_pridir = os.path.join(os.path.dirname(self.cfg.prior_weight), 'float_matrix.npy')
        if os.path.exists(float_matrix_in_pridir):
            logger.info('[LOAD] float_matrix from {}', float_matrix_in_pridir)
            float_matrix = torch.from_numpy(np.load(float_matrix_in_pridir)).float()
            self.n_class_loc = float_matrix.shape[1] 
            CHECK3D(float_matrix)
        else: 
            raise ValueError('loc_dist=cat require %s'%float_matrix_in_pridir)
        self.float_matrix = float_matrix 
        return self.float_matrix 

    # ---------------------------
    #    Forward Function       |
    # ---------------------------
    def decode(self, z):
        ''' vary loc 
        Args:
            z: latent dict {'sampled_loc', 'sampled_sel', 'sampled_stp', ...} 
                N = nloc 
                - z_cls: B,N,K: 1-of-K vector 
                - z_loc: B,N,2: range: -inf,+nan, need normalized
                - last dim is z_stp: B,N,1
        Returns: 
            pred: 
            canvas:
            latent (tensor): [B,nloc,2,canvas_size, canvas_size]
        '''
        pred, canv_dist = self.decoder(z) 
        canvas, latent = canv_dist['canvas'], canv_dist['latent'] 
        return pred, canvas, latent 
    
    # ---------------------------
    #   compute train loss fun  |
    # ---------------------------
    def loss_function(self, recon_x, x, latent_dict, canvas):
        '''Compute BCE and KLD for ELBO 
            pi_normed shape: B,nloc*nloc == latent_dim, K 
            KLD = q_y * log(q_y / (1 / prob_prior)) = q_y * (log(q_y) - log(prop_prior), )
        Args: 
            recon_x: 
            x:
            latent_dict: dict of latent: key: 'sampled_loc/sel/stp/extra_z' and 
                'logits_loc/sel/stp/extra_z': which is the parameters of different 
                latent variable 
            canvas: for vis 
        Returns: loss_dict: 
        '''
        loss, vis = {}, {}
        B = recon_x.shape[0]
        K, nloc = self.categorical_dim, self.nloci*self.nlocj
        BCE = self.compute_recont_loss(recon_x, x, B) 
        # == KL ==
        all_KL, kl_lossd, kl_visd = self.compute_KL(latent_dict, canvas) # , latent)
        loss.update(kl_lossd)
        vis.update(kl_visd)
        # == ELBO ==
        ELBO = self.cfg.kld_weight * all_KL + self.cfg.bce_weight * BCE 
        loss.update({'BCE':BCE, 'ELBO': ELBO })
        if self.xid is not None:
            assert(self.xid is not None) # require xid 
            xid = self.xid.to(x.device)
            reg_loss, reg_vis = self.compute_reg_loss(xid, latent_dict, B)
            loss.update(reg_loss)
            vis.update(reg_vis) 
        return loss, vis 

    def compute_KL(self, latent_dict, canvas): 
        '''
        Args: 
            latent_dict (dict) : {logits_sel, sampled_sel, sampled_stp,
                sampled_loc}
            canvas (tensor) 
        '''
        loss, vis = {}, {}
        B = canvas.shape[0]
        K, nloc = self.categorical_dim, self.nloci*self.nlocj
        assert(self.prior_model.name in ['rnn_one_head', 'cnn_head'])
        logits_sel = latent_dict['logits_sel'].view(B,nloc,K)  
        log_q_z1_x = torch.log_softmax(logits_sel, dim=2)
        all_KL = 0
        tic = time.time()
        if self.cfg.use_prior_model and self.cfg.kld_weight > 0:
            # -- prepare input for prior model -- 
            # B,nloc,ksel | B,nloc,kloc | B,nloc,1
            sampled_sel, sampled_loc, sampled_stp = latent_dict['sampled_sel'], \
                    latent_dict['sampled_loc'], latent_dict['sampled_stp'] 
            CHECKSIZE(sampled_sel, (B,nloc,K))
            CHECKSIZE(sampled_loc, (B,nloc,self.n_class_loc))
            CHECKSIZE(sampled_stp, (B,nloc,1))
            prior_input = torch.cat([sampled_sel, sampled_loc, sampled_stp], dim=2) 
            prior_output, prior_output_sampled = self.prior_model.evaluate(
                prior_input) 
            CHECKSIZE(prior_output, (B,nloc,K+self.n_class_loc+1))
            py_pid_BNK = prior_output[:,:,:K]
            log_p_z1 = torch.log_softmax(py_pid_BNK, dim=2) 
            
            CHECKTYPE(log_q_z1_x, log_p_z1)
            KLD_sel = log_q_z1_x.exp() * (-log_p_z1 + log_q_z1_x)
            KLD_sel = KLD_sel.view(B,nloc,K).sum(-1) 
            log_q_z2_x = torch.log_softmax(latent_dict['logits_loc'].view(
                B,nloc,self.n_class_loc), dim=2)
            py_loc_BNK = prior_output[:,:,K:K+self.n_class_loc]
            log_p_z2 = torch.log_softmax(py_loc_BNK, dim=2) # log(p_z_sel) 
            CHECKTYPE(log_q_z2_x, log_p_z2)
            KLD_loc = log_q_z2_x.exp() * (-log_p_z2 + log_q_z2_x)
            KLD_loc = KLD_loc.view(B,nloc,-1).sum(-1) 
            
            assert(self.add_stop)

            prob_stp = latent_dict['logits_stp']
            prob_stp = prob_stp.view(B,nloc)
            prior_valid_step = prior_output[:,:,-1]
            CHECKTYPE(prob_stp, prior_valid_step)
            eps = 1e-6 
            KLD_stp = prob_stp*(prob_stp/(prior_valid_step+eps) + eps).log() \
                + (1-prob_stp) * ((1-prob_stp)/(eps+1-prior_valid_step) + eps).log()
            KLD_stp = KLD_stp.sum(-1)
            all_KL += KLD_stp
            loss.update({'KLD_stp_print': KLD_stp})
            if latent_dict.get('sampled_stp') is not None: 
                sel_stp = latent_dict['sampled_stp']
                KLD_sel = KLD_sel * sel_stp.view(B,nloc) 
                KLD_loc = KLD_loc * sel_stp.view(B,nloc)
            KLD_sel = KLD_sel.sum(-1)
            KLD_loc = KLD_loc.sum(-1)  
        else:
            KLD_loc = logits_sel.new_zeros(B)
            KLD_sel = logits_sel.new_zeros(B)
        all_KL += KLD_loc+KLD_sel
        loss.update( { 'KLD_sel_print':KLD_sel, 'KLD_loc_print':KLD_loc, 'KLD': all_KL})
        return all_KL, loss, vis 

    def compute_reg_loss(self, xid, latent_dict, B): 
        ''' compute KL( p_h(z|x); q(z|x) ); called by loss_function  
        Args: 
            xid (tensor):
            locs_mu_2BN_01 (tensor): (2,B,nloc), range(0,1), predicted location; 
            log_q_z1_x (tensor): (B,nloc,K), 
                logsoftmax of K selection; predicted patch selection 
        Return:
            loss_dict, vis_dict
        '''
        loss, vis = {}, {}
        K,ph,nloc = self.categorical_dim, self.cfg.wh, self.latent_dim 
        prior_gt = self.prior_gt.index_select(0, xid) # B,N,5
        # 4D: [cls, loci, locj, mask]; 
        # ph_zcls:(B,nloc); ph_zloc_bn2_tl:(B,nloc,2); ph_mask:(B,nloc,1)
        ph_zcls, ph_zloc_bn2_tl, ph_mask = prior_gt[:,:,0],prior_gt[:,:,1:3], prior_gt[:,:,3:4]
        ph_zloc_bn2_tl = ph_zloc_bn2_tl.float() 
        ph_mask = ph_mask.float()
        # pid,loci,locj,has_gt_patch,has_gt_loc
        ph_mask_loc = ph_mask if prior_gt.shape[-1] == 4 else prior_gt[:,:,4:5] # B,N,1 
        ph_mask_loc = ph_mask_loc.float()
        ph_zcls = ph_zcls.float()
        
        if self.cfg.cat_vae.n_samples > 1 and xid.shape[0] != B: 
            size_g = self.prior_gt.shape[-1]
            B_ori = xid.shape[0] 
            prior_gt = prior_gt.view(1, B_ori, nloc, size_g).expand(
                    self.cfg.cat_vae.n_samples,-1,-1,-1).contiguous().view(B,nloc,size_g)
        ph_zloc_bn2_ctr = ph*0.5+ph_zloc_bn2_tl # (B,Nloc,2) range [0,28]
        # from (B,Nloc,2) to (B,Nloc,1) 
        ph_zloc_bn2_ctr_ind = self.loc_transform_f2i(ph_zloc_bn2_ctr.view(
            B*nloc,2)).view(B,nloc,-1).max(2)[1]
        logits_loc = latent_dict['logits_loc'] # B,Nloc,ncls_loc
        CHECKSIZE(logits_loc, (B,nloc,self.n_class_loc)) 
        loss['loc_loss'] = (F.nll_loss(
            torch.log_softmax( logits_loc.view(B*nloc,self.n_class_loc), dim=1), 
            ph_zloc_bn2_ctr_ind.view(B*nloc).long(), 
            ignore_index=-1, reduction='none')*ph_mask.view(B*nloc)).view(B,nloc).sum(-1) 
        # -- debug used --
        prob_stp = latent_dict.get('logits_stp')
        if prob_stp is not None: 
            valid_mask = (ph_zcls >= 0).float() # B,nloc; if gets 0: skip current step  
            ## ph_mask_loc expected to be same as valid_mask ? 
            stp_target = valid_mask.view(B,nloc)
            loss['stp_loss'] = F.binary_cross_entropy(prob_stp.view(B,nloc),
                stp_target, reduction='none').view(B,-1).sum(-1)
        logits_sel = latent_dict['logits_sel'].view(B,nloc,K)  
        log_q_z1_x = torch.log_softmax(logits_sel, dim=2)
        CHECKTYPE(loss['loc_loss'], ph_mask_loc, log_q_z1_x, ph_zcls)
        loss['sel_loss'] = (F.nll_loss(
            log_q_z1_x.view(B*nloc,K), ph_zcls.view(B*nloc).long(), 
            ignore_index=-1, reduction='none')*ph_mask.view(B*nloc)).view(B,nloc).sum(-1) 
        return loss, vis 

    # -----------------------------------------------
    # Eval Function: compute test elbo & vis        |
    # -----------------------------------------------
    @torch.no_grad()
    def test_loss(self, x, num_sample):
        '''
        Args: 
            x: input tensor, (B,C,H,W)
            num_sample: number of samples used to do iwae estimation of NLL 
        Returns:
            output: (dict), 
                {'pred': #pred img, shape(num_sample,B,C,H,W)
            out_d: (dict), {'NLL50', 'NLL1', 'NLL', 'BCE', 'KLD'}
        '''
        output = {} 
        B,K,N = x.shape[0], self.categorical_dim, self.N 
        B_new = num_sample * B
        x = x.view(1,B,-1).expand(num_sample, -1, -1).contiguous().view(B_new,-1) 

        # q(z|x), logits_loc: B,nloc,2 or B,nloc,3 
        estimator = 'stgs' if self.cfg.cat_vae.estimator == 'gs' else self.cfg.cat_vae.estimator 
        latent_dict = self.encode(x, estimator) # use stgs if 

        ## -- deal with cases where GT of some latent is given  --  
        te_sel_gt = self.cfg.vary_loc_vae.te_sel_gt
        if te_sel_gt:
            self.xid = self.xid.view(1,B).expand(num_sample,-1).contiguous().view(-1) 
        self.xid = None # set it as None to avoid computing loss 

        # x ~ p(z|x); decoder also sample and turn z_loc into z_stp 
        # TODO: add z_stp sampling ?? 
        pred, canvas, latent = self.decode(latent_dict) 

        # [test_loss] need to deal with prob_stp, to compute KL  
        loss, vis = self.loss_function(pred, x, latent_dict, canvas) #, latent) 
        output['pred']  = self.out2sampled(pred)
        BCE, KLD = loss['BCE'], loss['KLD']
        BCE = BCE.view(num_sample,B,-1).sum(-1).view(num_sample, B).double()
        KLD = KLD.view(num_sample,B,-1).sum(-1).view(num_sample, B).double()
        log_weight = (- BCE) - KLD   
        loss = - logmeanexp(log_weight, 0).mean()
        #loss = - helper.logmeanexp(log_weight, 0).mean()
        NLL  = loss  
        out_d = {'BCE':BCE.mean(), 'KLD':KLD.mean(), 'NLL': NLL}
        out_d[f'NLL{num_sample}'] = NLL 
        out_d[f'NLL1'] = - log_weight.mean() 
        return output, out_d 

    def encode(self, x, estimator):
        ''' x -> q(z|x), gives the sample function to the encoder, s.t. the encoder 
        can do sampling in the loop if needed 
        '''
        # sample z ~ q(z|x) 
        latent_dict = self.encoder(x, 
            sampler_sel=partial(draw_sample, estimator=estimator, temp=self.temp.detach()),
            sampler_stp=partial(draw_ber, temp=self.temp.detach()),
            sampler_loc=partial(draw_sample, estimator='stgs', temp=self.temp.detach())
            )
        return latent_dict 

    def fillq_gt_ifnid(self, latent_dict): 
        """
        fill latent_dict with hand-crafted gt if needed 
        Parameters
        ----------
        latent_dict : dict 
            with the sampled latent z
        Returns
        -------
        latent_dict : dict 
            updated latent_dict with the gt filled if needed
        """
        te_sel_gt = self.cfg.vary_loc_vae.te_sel_gt
        if te_sel_gt: 
            ## y_flat_gt: [B,Nloc,Ncls]; logits_loc_gt: [B,Nloc,Ncls]
            if te_sel_gt == 4: # need to do annealing, flip a coin to decide whether to use gt 
                y_flat_gt, logits_loc_gt = self.get_gt_sel_loc(self.xid) 
                nsample, nloc, _ = logits_loc_gt.shape
                # within test_loss, not using gt 
                coin = torch.zeros_like(self.xid) + self.temp_gt_q 
                coin = torch.bernoulli(coin).float().view(-1,1,1) # 1: use gt, 0: not use gt 
                sampled_loc_gt = self.loc_transform_f2i(
		    logits_loc_gt.view(nsample*nloc,2)*self.canvas_size).view(nsample, nloc, -1)
                latent_dict['sampled_loc'] = latent_dict['sampled_loc'] * (1-coin) + sampled_loc_gt * coin 
                latent_dict['sampled_sel'] = latent_dict['sampled_sel'] * (1-coin) + y_flat_gt * coin 
        return latent_dict


    @torch.no_grad() 
    def vis(self, x):
        ''' x -> encode q(z|x) -> sampling z -> decode p(x|z) 
        Args: x (tensor) 
        Returns: stack of vis (tensor)
        '''
        img_size = self.img_size 
        Hc,Wc = self.canvas_size,self.canvas_size 
        B,N,K = x.shape[0],self.N, self.K
        estimator = 'stgs' if self.cfg.cat_vae.estimator == 'gs' else self.cfg.cat_vae.estimator 
        latent_dict = self.encode(x, estimator)
        ## -- deal with cases where GT of some latent is given  -- 
        self.xid = self.xid.view(B) 
        latent_dict = self.fillq_gt_ifnid(latent_dict)

        pred, canvas, latent = self.decode(latent_dict)
        pred_out = self.out2sampled(pred).view(B,self.imgd,self.img_size,self.img_size)

        # B,1,H*ph,W*pw
        latent = latent.view(B,-1,2,self.canvasd,Hc,Wc) 
        nloc = latent.shape[1]
        # -- before debuggging -- 
        latent_vis = latent.view(-1,2*self.canvasd,Hc,Wc)
        latent_vis = F.interpolate(latent_vis, img_size, mode='bilinear', 
            align_corners=True).view(B,-1,2,self.canvasd,img_size,img_size) # upsample 
        canvas_vis = canvas.view(-1,self.canvasd,Hc,Wc)
        canvas_vis = F.interpolate(canvas_vis, img_size, mode='bilinear', 
            align_corners=True).view(B,self.canvasd,img_size,img_size) # upsample 
        vis_sp   = [B,self.imgd,   img_size,img_size]
        vis_canv = [B,self.canvasd,img_size,img_size]
        
        output = torch.cat([ 
            latent_vis[:,:,1].max(1)[0].view(*vis_canv).expand(*vis_sp), 
            canvas_vis.view(*vis_canv).expand(*vis_sp),
            pred_out.view(*vis_sp) 
            ]) 
        return output.view(-1,B,self.imgd,img_size,img_size)

    def compute_elbo(self, x):
        ''' x -> encoder q(z|x) -> sampling z -> compute elbo 
        Args: x (tensor) 
        Returns: elbo (value)
        '''
        ### need to tile logits_loc 
        b,n,k = x.shape[0],self.N, self.K
        latent_dict = self.encode(x, self.cfg.cat_vae.estimator)
        ## -- deal with cases where GT of some latent is given  -- 
        self.xid = self.xid.view(b)
        latent_dict = self.fillq_gt_ifnid(latent_dict)
        # For evaluation, we compute analytical KL, otherwise we cannot compare
        # get x ~ p(x|z_cls, z_loc, z_stp)
        B,N,K = x.shape[0],self.N, self.K
        pred, canvas, latent = self.decode(latent_dict) 
        # B,1,H*ph,W*pw
        loss, vis = self.loss_function(pred, x, latent_dict, canvas) ## , latent) 
        return loss, vis 
    
    # ----------------------
    #   sample function    |
    # ----------------------
    @torch.no_grad()  
    def sample_from_prior(self, B):
        ''' 
        Returns: 
            z (tensor): (B,canv_d,csize,csize)
        '''
        img_size = self.img_size
        csize = self.canvas_size
        upscale = lambda x: F.interpolate(x.view(B,1,csize,csize), img_size) 
        # sample: B,Nloc,canvas_size,canvas_size 
        if self.prior_model.name == 'rnn_one_head' or self.prior_model.name == 'cnn_head':
            sample = self.prior_model.generate(
                shape=self.nloci*self.nlocj, batch_size=B) #64 
            z = {'sampled_sel': sample[:,:,:self.categorical_dim], 
                 'sampled_loc': sample[:,:,self.categorical_dim:-1], 
                 'sampled_stp': sample[:,:,-1].unsqueeze(2)}
            canvas = None
        else:
            sample, loc_map = self.prior_model.generate(
                shape=self.nloci*self.nlocj, batch_size=B) #64 
            sample = sample.view(B,self.nloci*self.nlocj,csize,csize)[:,-1:,:,:] 
            # B,1,canvas_size,canvas_size
            if self.canv_d == 2:
                loc_m = loc_map.view(B,self.nloci*self.nlocj,csize,csize).max(1)[0].unsqueeze(1) #[:,-1:,:,:]
                z = torch.cat([sample, loc_m], dim=1) 
                canvas = [sample, loc_m]  
                if csize < img_size: 
                    canvas = [upscale(c) for c in canvas]
            if self.canv_d == 1:
                z = sample 
                canvas = sample if csize == img_size else upscale(sample) 
        return z, canvas  

    @torch.no_grad()  
    def sample(self, nsample=64): 
        ''' draw samples: x ~ p(x|z), z ~ p(z) '''
        B = nsample
        K = self.categorical_dim
        nsample_z = B 
        # -- different way to draw sample, depends on the prior model  --  

        if self.prior_model.name in ['rnn_one_head', 'cnn_head'] :
            latent_dict, _ = self.sample_from_prior(nsample_z)
            pred, canvas, _ = self.decode(latent_dict) 
        else:
            sample, canvas = self.sample_from_prior(nsample_z)
            CHECKSIZE(sample, (B,1,self.canvas_size,self.canvas_size))
            sshape = sample.shape[1:] # canv_d,csize,csize
            if nsample_z < B:
                sample = sample.view(1,nsample_z,-1).expand(B//nsample_z,-1,-1
                        ).contiguous().view(B,*sshape) # View as B,cd,cs,cs 
            S = B  
            pred, canvas = self.decoder.forward_with_canvas(
                        sample, return_canvas=True)
            canvas = canvas['canvas']
        pred = self.out2sampled(pred).view(B,self.imgd,self.img_size,self.img_size)
        return pred, canvas

    def get_optim(self, lr):
        logger.info('*'*30)
        logger.info('[build_optimizer] diff_lr: %d | '%(
            self.cfg.optim.diff_lr))
        logger.info('*'*30)
        other_param, enc_param = [],[]
        head_loc = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'prior_model' in name:
                    logger.info('found prior_model weight: {}', name)
                    continue 
                if 'encoder' in name:
                    enc_param.append(param) 
                else:
                    other_param.append(param)
        diff_lr = self.cfg.optim.diff_lr 
        if diff_lr == 0:
            lr_other = lr 
        else:
            lr_other = 0.1 / diff_lr * lr 
        logger.info('LR for enc: {}, LR for other: {}', lr, lr_other)
        # lr_other = 0.1*lr if self.cfg.optim.diff_lr else lr 
        return optim.Adam([
            {'params': enc_param,   'lr': lr}, 
            {'params': other_param, 'lr': lr_other}],
            lr=lr)

    def set_bank(self, device):
        info_dict = self.info_dict
        self.patch_bank, self.prior_model, self.prior_gt = load_prior_with_patch_bank(
            info_dict=info_dict, cfg=self.cfg, device=device, metric=self.metric, n_class_loc=self.n_class_loc) 
        float_matrix = self.float_matrix 
        patch_bank = self.patch_bank
        self.grid_H   = torch.arange(1., 1.+self.canvas_size).to(device).detach()
        self.zcls_BHWK_to_psel_BHW_phw = partial(zcls_BHWK_to_psel_BHW_phw, 
                info_dict=info_dict, patch_bank=patch_bank)
        self.coords_ij_to_distmap = partial(coords_ij_to_distmap,
                info_dict=info_dict, grid_H=self.grid_H)
        self.transform_ij = partial(transform_ij, 
                info_dict=info_dict, grid_H=self.grid_H)
        self.get_gt_sel_loc = partial(get_gt_sel_loc, 
                info_dict, self.prior_gt)
        self.vis_prior_output = 0
        float_matrix = float_matrix if float_matrix is not None else \
            ph_helper.prepare_loc_float_matrix(self.canvas_size, 
                self.cfg.vary_loc_vae.loc_stride)
        self.loc_transform_i2f = partial(ph_helper.loc_transform_i2f, float_matrix)
        self.loc_transform_f2i = partial(ph_helper.loc_transform_f2i, float_matrix)
        logger.info('[INIT] float_matrix: {}', float_matrix.shape)
        # -- set funct for decoder -- 
        self.set_dec_func(
            self.zcls_BHWK_to_psel_BHW_phw,
                self.coords_ij_to_distmap, self.transform_ij, 
                self.loc_transform_i2f, self.loc_transform_f2i)
        cfg = self.cfg 
        self.canvas_plotter = CanvasPlotter(n_class_loc=self.n_class_loc, n_class_sel=cfg.K,
            nloc=cfg.vary_loc_vae.nloc, dataset=cfg.dataset, 
            loc_dist='cat', device=device, 
            patch_bank=self.patch_bank, patch_size=cfg.ww, ##patch_size, 
            loc_stride=cfg.vary_loc_vae.loc_stride, 
            add_stop=cfg.cat_vae.add_stop, 
            loc_transform_i2f=self.loc_transform_i2f,
            loc_transform_f2i=self.loc_transform_f2i)
        if self.prior_model.name == 'cnn_head':
            logger.info('set plotter')
            self.prior_model.canvas_plotter = self.canvas_plotter

    def set_dec_func(self, *args):
        self.decoder.set_func(*args)
