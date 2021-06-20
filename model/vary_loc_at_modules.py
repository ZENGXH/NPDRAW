import numpy as np
from loguru import logger
import torch
from torch import nn 
from torch.nn import functional as F
from utils import data_helper, ph_helper
from utils.checker import *
from model.modules import flatten, View, Interpolate, Residual, Permute, get_conv_encoder 
from torchvision.models.resnet import BasicBlock
from model.modules import get_prepost_pross

# Sample from a Gaussian distribution
class PosteriorModule(nn.Module):
    def __init__(self, cfg, n_class_loc, preproc): 
        """ cfg: config 
        preproc (list): modules serve for prior 
        enc_version: 
            0: mlp head, 4 downsample block 
            1: fcn head, 4 downsample block
            2: fcn head, 1 downsample block + conv_encoder (same as fixed-loc) 
        """
        super().__init__()
        self.cfg = cfg 
        self.n_class_loc = n_class_loc 
        self.fcn = cfg.vary_loc_vae.enc_version > 0 ## if version == 1: use fcn 
        self.loc_var = None
        self.add_stop = cfg.cat_vae.add_stop
        self.latent_dim = cfg.vary_loc_vae.nloc 
        self.loc_dist = 'cat' 
        self.canvas_size = data_helper.get_canvsize(cfg.dataset)

        self.enc_hid = enc_hid = cfg.enc_hid
        self.n_class_sel = cfg.K 

        block = lambda x,y: [ nn.Conv2d(x,y,3,2,1), nn.BatchNorm2d(enc_hid), nn.ReLU()] 
        imgd = data_helper.get_imgd(cfg.dataset)
        self.img_size = data_helper.get_imgsize(cfg.dataset) 
        if cfg.vary_loc_vae.enc_version == 2: 
            enc_layer = [*block(imgd, enc_hid), *get_conv_encoder(enc_hid, enc_hid, 1, enc_hid)]
        else:
            enc_layer = [*block(imgd, enc_hid), *block(enc_hid,enc_hid), 
                *block(enc_hid,enc_hid), *block(enc_hid, enc_hid)] # 14->7->4->2

        if self.fcn:
            # size of the feature map 
            hw = int(np.sqrt(self.latent_dim))
            assert(hw**2 == self.latent_dim), 'latent_dim required to be n^2, get: %d'%self.latent_dim 
            enc_layer.append(Interpolate((hw,hw), mode='bilinear')) 
            enc_layer.extend([nn.Conv2d(enc_hid,enc_hid,1,1), nn.BatchNorm2d(enc_hid), nn.ReLU()])
        else:
            if self.img_size > 28: 
                enc_layer.append(Interpolate((2,2), mode='bilinear'))
            enc_layer.extend([flatten(), nn.Linear(4*enc_hid, enc_hid), nn.ReLU()])
        self.embed = nn.Sequential( *preproc, *enc_layer )
        
        self.categorical_dim = self.n_class_sel 
        self.build_head()

    def build_head(self): 
        enc_hid = self.enc_hid
        latent_dim = self.latent_dim
        logger.info('[Build Encoder]: n_class={} | latent_dim={} | out dim={}', 
                self.n_class_loc, latent_dim, self.n_class_loc*latent_dim)
        if not self.fcn:
            linear_head = lambda enc_hid, n_class:nn.Sequential(nn.Linear(enc_hid, enc_hid), nn.ReLU(), 
                nn.Linear(enc_hid, n_class*latent_dim))
            self.head_cls = linear_head(enc_hid, self.n_class_sel) 
            self.head_stop = nn.Sequential(nn.Linear(enc_hid, latent_dim), nn.Sigmoid())
            self.head_loc = linear_head(enc_hid, self.n_class_loc) 
        else: # fcn
            nloc = int(np.sqrt(latent_dim))
            conv_head = lambda enc_hid, n_class: nn.Sequential(
                    nn.Conv2d(enc_hid, n_class, 1, 1), Permute((0,2,3,1) # B,D,H,W->B.H.W.D
                    ))
            ## raise NotImplementedError
            self.head_cls = conv_head(enc_hid, self.n_class_sel) 
            self.head_stop = nn.Sequential(conv_head(enc_hid, 1), nn.Sigmoid())
            self.head_loc = conv_head(enc_hid, self.n_class_loc) 
             
    def forward(self, x, sampler_sel, sampler_loc, sampler_stp):
        '''
        Args: x_data, images 
        Return: 
            h3_cls: (B,nloc,K), patch selection 
            h3_loc: (B,nloc,2) or (B,nloc,3), if add_stop 
        '''
        latent = {}
        B = x.shape[0]
        h2 = self.embed(x)
        # -- q(z_cls|x) -- 
        h3_cls = self.head_cls(h2).view(B, self.latent_dim, self.categorical_dim)  
        latent['logits_sel'] = h3_cls
        ## if sampler is not None:
        latent['sampled_sel'] = sampler_sel(h3_cls.view(B*self.latent_dim,self.categorical_dim)).view(
            B,self.latent_dim,self.categorical_dim)
        # -- q(z_loc|x) -- 
        out_loc = self.head_loc(h2)
        logits_loc = out_loc.view(B,self.latent_dim,self.n_class_loc) 
        latent.update({'logits_loc': logits_loc})
        if sampler_loc is not None:
            latent['sampled_loc'] = sampler_loc(logits_loc.view(-1, self.n_class_loc)).view(
                B,self.latent_dim,self.n_class_loc)

        if self.add_stop:
            prob_stp = self.head_stop(h2).view(B,self.latent_dim,1)
            latent['logits_stp']  = prob_stp
            sel_stp = torch.bernoulli(prob_stp).float()  # stochastically binarize
            latent['sampled_stp'] = sel_stp 
        return latent
# Generative Network
class GeneratorWithPB(nn.Module):
    fcn = 0
    def __init__(self, cfg, postproc, generative_pxh, n_class_loc): 
        ''' generative_pxh: nn.Sequential of conv layers; take canvas as input, output the 
            prediction, including the postproc 
        Parameters:
        -----------
        cfg : config 
        postproc : ignored 

        '''
        super().__init__()
        self.overlap = 'max' 
        self.cfg = cfg 
        self.loc_dist = 'cat' 
        self.add_stop = 1 
        self.canvas_size = data_helper.get_canvsize(cfg.dataset)
        self.img_size = data_helper.get_imgsize(cfg.dataset) 
        self.bg_mean = 0
        self.canvasd = data_helper.get_canvasd(cfg)
        self.n_class_loc = n_class_loc  
        # n_row * n_col 
        logger.info('[Generator] canv_size={}, n_class_loc={}, overlap={}', 
            self.canvasd, self.n_class_loc,  self.overlap) 
        self.nloc = self.nloci = cfg.vary_loc_vae.nloc
        self.nlocj = 1  
        latent_dim = self.nloci*self.nlocj 
        self.categorical_dim = self.cfg.K
        canv_da = self.canvasd
        dec_hid = cfg.dec_hid
        assert(not self.fcn) 

        # p(x|z)
        x_dim = self.canvas_size**2 * self.canvasd
        self.emb_canv = None 
        self.generative_pxh = generative_pxh
        # assert(not cfg.cat_vae.use_mlp_dec)
        assert(self.emb_canv is None)

    def forward(self, c):  
        '''
        Return:
            recont: p(x|z)
            canvas: canv(z_loc, z_sel, z_stp)
            latent: [B,nloc,2,cs,cs]; with loc_map, pat_map
        '''
        canvas, latent = self.create_canvas(c)  
        # assert(self.patch_bank is not None), 'require set_pb before run '
        h = canvas # same 
        recont = self.generative_pxh(h)
        return recont, {'canvas': canvas, 'latent':latent} 

    def forward_with_canvas(self, canvas, extra_latent=None, return_canvas=False):
        h = canvas 
        recont = self.generative_pxh(h)
        if return_canvas: 
            return recont, {'canvas': canvas} 
        else:
            return recont

    def create_canvas(self, z): 
        ''' 
        Args: 
            z: dict of z sampled from q(z|x) 
        Return:
            z shape: (B,canvasd,canv_size,canvas_size)
            latent: (B,nloc,2,canvasd,cs,cs)
        '''
        z_cls, z_loc, z_stp = z['sampled_sel'], z['sampled_loc'], z['sampled_stp']
        B = z_cls.shape[0]
        nloc, img_size, canvas_size = self.nloc, self.img_size, self.canvas_size
        Hc = Wc = canvas_size 
        K,ph = self.categorical_dim, self.cfg.wh
        CHECKSIZE(z_cls, (B,nloc,K))

        k_loc = self.n_class_loc
        CHECKSIZE(z_loc, (B,nloc,k_loc))
        # -- transform from categorical_dim --
        z_loc  = self.loc_transform_i2f(z_loc.view(B*nloc,k_loc)).view(B,nloc,2) 
        g_i = z_loc[:,:,0].view(B*nloc,1)  
        g_j = z_loc[:,:,1].view(B*nloc,1)  
        psel  = self.zcls_BHWK_to_psel_BHW_phw(z_cls.unsqueeze(2)) # BNloc,ps,ps,canvasd

        # center location of the patches 
        map_shape     = [B, nloc, canvas_size, canvas_size, self.canvasd]
        map_shape_loc = [B, nloc, canvas_size, canvas_size, 1]
        ## logger.info('g_i: {}', g_i[0])
        loc_map = self.coords_ij_to_distmap(g_i.view(B,nloc), g_j.view(B,nloc)).view(*map_shape_loc).expand(*map_shape)
        pat_map = self.transform_ij(g_i,g_j,psel).view(*map_shape)
        
        if self.overlap == 'max':
            ps = psel.shape[1] 
            step_map = torch.arange(1.,nloc+1).view(1,nloc,1,1,1).to(
                loc_map.device).detach().expand(B,-1,ps,ps,self.canvasd).reshape(psel.shape)
            pat_map_fg = self.transform_ij(g_i,g_j,step_map).view(*map_shape).detach() # BNloc,canvas_size,canvas_size,canvasd
        
            if self.add_stop:
                CHECKSIZE(z_stp, (B,nloc,1))
                # loc_map: B,nloc,csize,csize 
                z_stp = z_stp.view(B,nloc,1,1,1)
                loc_map = loc_map * z_stp
                pat_map = pat_map * z_stp
                pat_map_fg = pat_map_fg * z_stp 

            canvas    = pat_map.max(1)[0].view(B,Hc,Wc,self.canvasd).permute(0,3,1,2).contiguous() # select the max value over Nsteps 
            canvas_fg = pat_map_fg.sum(1).view(B,Hc,Wc,self.canvasd).permute(0,3,1,2).contiguous() # select the fg over Nsteps, and the rest is bg 
        else:
            raise ValueError('Not suppoer overlap=%s'%self.overlap)
        # get pasted patch
        # shape: B,nloc,2,canvas_size,canvas_size,canvasd
        latent = torch.cat([pat_map.unsqueeze(2), loc_map.view(B,nloc,1,Hc,Wc,self.canvasd)],dim=2) 
        latent = latent.permute(0,1,2,5,3,4).contiguous() # B.Nloc.2.H.W.D -> B.Nloc.2.D.H.W 
        z = canvas.view(B, -1) # output for plot? 
        z = z.view(B, self.canvasd, Hc, Wc) 
        assert(not self.canvasd == 2)
        return z, latent 

    def set_func(self,zcls_BHWK_to_psel_BHW_phw,coords_ij_to_distmap,transform_ij,
            loc_transform_i2f, loc_transform_f2i):
        '''
            coords_ij_to_distmap: func, def in model.transform 
        '''
        self.zcls_BHWK_to_psel_BHW_phw=zcls_BHWK_to_psel_BHW_phw 
        self.coords_ij_to_distmap=coords_ij_to_distmap 
        self.transform_ij=transform_ij 
        self.loc_transform_i2f, self.loc_transform_f2i = loc_transform_i2f, loc_transform_f2i

def build_model(cfg, n_class_loc):
    """ build enc and dec for vary-loc 
    Parameters
    ----------
    cfg : config, 
    n_class_loc : int 
        N loc class after quantization 

    Returns 
    -------
    encoder : PosteriorModule 
    decoder : GeneratorWithPB
    """
    # follow the embed, head, fc5 in model/gumbel_vae_vary_loc.py model 
    img_size = data_helper.get_imgsize(cfg.dataset)
    imgd = data_helper.get_imgd(cfg.dataset)
    dec_hid = cfg.dec_hid 
    enc_hid = cfg.enc_hid 
    pix_class = data_helper.get_pixel_nclass(cfg.dataset)
    out_dist = data_helper.get_out_dist(cfg.dataset)
    out_dist_npara = data_helper.get_out_dist_npara(out_dist, pix_class)
    # -- set process: pre and post -- 
    preproc, postproc = get_prepost_pross(img_size, out_dist_npara, 
        imgd, pix_class, dec_hid, cfg, out_dist) # map hid -> img_size 
         
    preproc.append(View((imgd,img_size,img_size))) # since we use conv layer ? 

    assert( 2 == cfg.vary_loc_vae.stridep), 'not support larger stride ' 
    # -- build decoder -- 
    ker = 4 # ker = 4 ensure the up sample is matched
    pad = 1 
    tker = 4 
    tpad = 1 
    gen_list = []
    canv_d = data_helper.get_canvasd(cfg) # cfg.vary_loc_vae.canv_d 

    canvas_size = data_helper.get_canvsize(cfg.dataset)

    gen_list.extend([nn.Conv2d(canv_d,dec_hid,ker,2,pad,bias=False),  nn.BatchNorm2d(dec_hid), nn.ReLU()])
    gen_list.extend([nn.Conv2d(dec_hid, dec_hid,ker,2,pad,bias=False), nn.BatchNorm2d(dec_hid), nn.ReLU()])
    gen_list.append(Residual(dec_hid)) 
    gen_list.append(Residual(dec_hid)) 
    for i in range(2): 
        gen_list.extend(
           [nn.ConvTranspose2d(dec_hid, dec_hid, tker, 2, tpad, bias=False), 
            nn.BatchNorm2d(dec_hid), nn.ReLU()])
    generative_pxh = nn.Sequential( *gen_list, *postproc )
    encoder = PosteriorModule(cfg, n_class_loc, preproc=preproc)
    decoder = GeneratorWithPB(cfg, postproc, 
        generative_pxh=generative_pxh, n_class_loc=n_class_loc)
    return encoder, decoder 
