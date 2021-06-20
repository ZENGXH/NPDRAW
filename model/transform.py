import torch
import utils 
from loguru import logger 
from utils.checker import * 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils import data_helper, ph_helper


class CanvasPlotter(object):
    '''
    one class hold several function within transform file; 
    '''
    def __init__(self, n_class_loc, n_class_sel, nloc, dataset, 
            loc_dist, device, patch_bank, patch_size, add_stop, loc_stride,
            loc_transform_i2f, loc_transform_f2i, use_bg_mean=False
            ):  
        '''
            patch_bank in shape [1,nH,nW,K,ph,pw,imgd]
        '''
        CHECK7D(patch_bank)
        self.add_stop = add_stop
        # assert(self.add_stop), 'requried'
        self.loc_stride = loc_stride
        self.fcn = 0 
        self.loc_dist = loc_dist 
        self.n_class_loc = n_class_loc 
        self.n_class_sel = n_class_sel 
        
        self.img_size = data_helper.get_imgsize(dataset)
        self.canvas_size = data_helper.get_canvsize(dataset)
        self.imgd = data_helper.get_imgd(dataset) 
        if use_bg_mean:
            self.bg_mean = data_helper.get_img_mean(dataset)
        self.use_bg_mean = use_bg_mean

        self.nloc = nloc 
        self.categorical_dim = n_class_sel 
        self.patch_size = patch_size
        self.device = torch.device(device)
        logger.info('[INIT CanvasPlotter]: use_bg_mean={}, img_size={}, nloc={}, canvas_size={}, imgd={}, patch_size={}',
            self.use_bg_mean, self.img_size, self.nloc, self.canvas_size, self.imgd, self.patch_size)

        #if self.loc_dist == 'Gaussian' and self.fcn:
        #    self.loc_discrete = 0 
        #    self.loc_reg_from_anchor = 1
        #elif self.loc_dist == 'Gaussian': 
        #    self.loc_discrete = self.loc_reg_from_anchor = 0
         
        if self.loc_dist == 'cat': 
            self.loc_discrete = 1 
            self.loc_reg_from_anchor = 0
            assert(self.loc_stride), 'require loc_stride value'
            # -- location transform -- 
            # float_matrix = ph_helper.prepare_loc_float_matrix( 
            #    self.canvas_size, self.loc_stride)
            self.loc_transform_i2f = loc_transform_i2f #partial(ph_helper.loc_transform_i2f, float_matrix)
            self.loc_transform_f2i = loc_transform_f2i #partial(ph_helper.loc_transform_f2i, float_matrix)
        else:
            raise ValueError('not support loc_dist: %s'%self.loc_dist) 
        self.info_dict = {'categorical_dim': self.categorical_dim, 
                'nloci': self.nloc,
                'nlocj':1, 'ph': self.patch_size, 'pw':self.patch_size, 
                'canvas_size':self.canvas_size, 'img_size':self.img_size, 
                ### 'latent_dim': self.latent_dim,  
                'imgd':self.imgd, 'canvasd': self.imgd
                }
        self.canvasd = self.imgd
        self.grid_H   = torch.arange(1., 1.+self.canvas_size).to(device).detach()
        self.zcls_BHWK_to_psel_BHW_phw = partial(zcls_BHWK_to_psel_BHW_phw, 
                info_dict=self.info_dict, patch_bank=patch_bank)
        self.coords_ij_to_distmap = partial(coords_ij_to_distmap,
                info_dict=self.info_dict, grid_H=self.grid_H)
        self.transform_ij = partial(transform_ij, 
                info_dict=self.info_dict, grid_H=self.grid_H)
    

    def vis_gen_process(self, z, break_in_step=True, customize_nloc=0):
        """ visulize the how generation process, with bounding box draw on canvas 
        Args: 
            z: dict of z sampled from q(z|x) 
            sampled_sel: B.nloc.K1, can be soft 
            sampled_loc: B.nloc.K2, one-hot | top-left or center ??  
            sampled_stp: B.nloc.1 #? 
            customize_nloc: support taking different length of seq as input 
        """
        z_cls, z_loc, z_stp = z['sampled_sel'], z['sampled_loc'], z['sampled_stp']
        B = z_cls.shape[0]
        nloc, img_size, canvas_size = self.nloc, self.img_size, self.canvas_size
        if customize_nloc: 
            nloc = customize_nloc
        Hc = Wc = canvas_size 
        K,ph = self.n_class_sel, self.patch_size
        CHECKSIZE(z_cls, (B,nloc,K))
        # -- decode location output to (x,y) -- 
        assert(not self.loc_reg_from_anchor and self.loc_discrete)
        k_loc = self.n_class_loc
        CHECKSIZE(z_loc, (B,nloc,k_loc))
        # -- transform from categorical_dim --
        z_loc  = self.loc_transform_i2f(z_loc.view(B*nloc,k_loc)).view(B,nloc,2) 
        g_i = z_loc[:,:,0].view(B*nloc,1)
        g_j = z_loc[:,:,1].view(B*nloc,1)  
        # map to location; [0,Wc], [0,Hc] 
        if customize_nloc:
            psel  = self.zcls_BHWK_to_psel_BHW_phw(z_cls.unsqueeze(2), nloc=customize_nloc) # B,N,1,K
        else:
            psel  = self.zcls_BHWK_to_psel_BHW_phw(z_cls.unsqueeze(2)) # B,N,1,K

        # center location of the patches 
        map_shape = [B,nloc,canvas_size, canvas_size, self.imgd]
        pat_map = self.transform_ij(g_i, g_j, psel).view(*map_shape).cpu().numpy() 
        out_list = []
        if break_in_step: 
            for b in range(B):
                for locid in range(nloc):
                    img = (pat_map[b,locid] * 255).astype(np.uint8) # canvas_size, canv_size, imgid
                    img = Image.from_numpy(img)
                    draw = ImageDraw.Draw(img)
                    ri_color = tuple([0,255,255])
                    ti = int(g_i[b*nloc+locid] - ph*0.5)
                    tj = int(g_j[b*nloc+locid] - ph*0.5)
                    draw.rectangle([ (tj,ti), (tj+ph,ti+ph)], outline=ri_color)
                    out_list.append(img)
        else:
            for b in range(B):
                img = (pat_map[b,:].max(0)[0] * 255).astype(np.uint8) # canvas_size, canv_size, imgid
                for locid in range(nloc):
                    img = (pat_map[b,locid] * 255).astype(np.uint8) # canvas_size, canv_size, imgid



    @logger.catch(reraise=True)
    def create_canvas(self, z, break_in_step=False, customize_nloc=0, return_per_step_loc=0): 
        ''' 
        Args: 
            z: dict of z sampled from q(z|x) 
            sampled_sel: B.nloc.K1, can be soft 
            sampled_loc: B.nloc.K2, one-hot | top-left or center ??  
            sampled_stp: B.nloc.1 #? 
            customize_nloc: support taking different length of seq as input 
        Return:
            z shape: (B,1,canv_size,canvas_size)
            latent: (B,nloc,2,cs,cs)
        if break_in_step: 
        return z: (B*nloc,D,canvas_size,canvas_size)
            plot the canvas being plot progressively 
        '''
        z_cls, z_loc, z_stp = z['sampled_sel'], z['sampled_loc'], z['sampled_stp']
        B = z_cls.shape[0]
        nloc, img_size, canvas_size = self.nloc, self.img_size, self.canvas_size
        if customize_nloc: 
            nloc = customize_nloc
        Hc = Wc = canvas_size 
        K,ph = self.n_class_sel, self.patch_size
        CHECKSIZE(z_cls, (B,nloc,K))

        if self.loc_reg_from_anchor: 
            CHECKSIZE(z_loc, (B,nloc,2))
            z_loc  = self.loc_trans_anchor(z_loc) 
            # map to location; [0,Wc], [0,Hc] 
        elif self.loc_discrete:
            k_loc = self.n_class_loc
            CHECKSIZE(z_loc, (B,nloc,k_loc))
            # -- transform from categorical_dim --
            z_loc  = self.loc_transform_i2f(z_loc.view(B*nloc,k_loc)).view(B,nloc,2) 
            # map to location; [0,Wc], [0,Hc] 
        else:
            CHECKSIZE(z_loc, (B,nloc,2))
            # map to location; [0,Wc], [0,Hc] 
        g_i = z_loc[:,:,0].view(B*nloc,1)
        g_j = z_loc[:,:,1].view(B*nloc,1)  
        if customize_nloc:
            psel  = self.zcls_BHWK_to_psel_BHW_phw(z_cls.unsqueeze(2), nloc=customize_nloc) # B,N,1,K
        else:
            psel  = self.zcls_BHWK_to_psel_BHW_phw(z_cls.unsqueeze(2)) # B,N,1,K

        # center location of the patches 
        map_shape = [B,nloc,canvas_size, canvas_size, self.imgd]
        pat_map = self.transform_ij(g_i, g_j, psel).view(*map_shape)
        if self.add_stop:
            CHECKSIZE(z_stp, (B,nloc,1))
            # loc_map: B,nloc,csize,csize 
            z_stp = z_stp.view(B,nloc,1,1,1)
            ## loc_map = loc_map * z_stp
            pat_map = pat_map * z_stp
        # B,nloc,Hc,Wc,D -> (max over the nloc) -> B,Hc,Wc,D -> (permute) -> B,D,Hc,Wc
        if not break_in_step:
            canvas   = pat_map.max(1)[0].view(B,Hc,Wc,self.imgd).permute(0,3,1,2).contiguous() # select the max value over Nsteps 
        else:
            loc_map = self.transform_ij(g_i, g_j, psel*0+1).view(*map_shape)[...,0] # only select the first one C
            #B,nloc,cs,cs,D
            canvas = []
            loc_map_list = []
            for locid in range(nloc):
                canvas.append(pat_map[:,:locid+1].max(1)[0].view(B,Hc,Wc,self.imgd).permute(0,3,1,2).contiguous())
                loc_map_list.append(loc_map[:,:locid+1].max(1)[0].view(B,1,Hc,Wc))
            canvas = torch.stack(canvas, dim=1)
            loc_map_list = torch.cat(loc_map_list, dim=1)
            if not return_per_step_loc: 
                return canvas.view(B*nloc,self.imgd,Hc,Wc),loc_map_list.view(B*nloc,1,Hc,Wc) 
            else:
                return canvas.view(B*nloc,self.imgd,Hc,Wc),loc_map.view(B*nloc, Hc,Wc,self.imgd)
        # get pasted patch
        # shape: B,nloc,2,canvas_size, canvas_size
        #logger.info('pat_map: {}, loc_map: {}', pat_map.shape, loc_map.shape)
        ## latent = torch.cat([pat_map, loc_map.view(B,nloc,1,Hc,Wc)],dim=2)
        z = canvas.view(B, -1) # output for plot? 
        z = z.view(B, self.imgd, Hc, Wc) 
        #logger.info('output: {}', latent.shape)
        return z ## , latent 


def zcls_BHWK_to_psel_BHW_phw(z_cls, info_dict, patch_bank, nloc=None): #, z_cls):
    ''' obtain selected patches 
    Args: 
        z_cls (tensor): B,H,W,K output of ? 
    Returns:
        psel (tensor): B,H,W,K,ph,pw,canvasd -> BHW,ph,pw,canvasd
    '''
    B = z_cls.shape[0]
    categorical_dim, nloci, nlocj, ph, pw, canvasd = info_dict['categorical_dim'], info_dict['nloci'], \
            info_dict['nlocj'], info_dict['ph'], info_dict['pw'], info_dict['canvasd'] 
    if nloc: 
        nloci = nloc 
        nlocj = 1
        K,H,W = categorical_dim, nloci, nlocj 
        CHECKSIZE(z_cls, (B,H,W,K))
        # logger.info('check selectk: {}', z_cls.mean(3).view(-1)) 
        patch_bank = patch_bank[:,:nloc]
        CHECKSIZE(patch_bank, (1,H,W,K,ph,pw,canvasd))
        target_shape = [B,H,W,K,ph,pw,canvasd]
        z_cls = z_cls.view(B,H,W,K,1,1,1).expand(*target_shape) #-1,-1,-1,-1,ph,pw,canvasd) # B,nH,nW,K,ph,pw 
        psel = z_cls * patch_bank.expand(*target_shape) #B,-1,-1,-1,-1,-1) # B,nh,nw,K,ph,pw
        psel_sum = psel.sum(3).view(B*H*W,ph,pw,canvasd) # sum along K dimension  

    else:
        K,H,W = categorical_dim, nloci, nlocj 
        CHECKSIZE(z_cls, (B,H,W,K))
        # logger.info('check selectk: {}', z_cls.mean(3).view(-1)) 
        CHECKSIZE(patch_bank, (1,H,W,K,ph,pw,canvasd))
        target_shape = [B,H,W,K,ph,pw,canvasd]
        z_cls = z_cls.view(B,H,W,K,1,1,1).expand(*target_shape) #-1,-1,-1,-1,ph,pw,canvasd) # B,nH,nW,K,ph,pw 
        psel = z_cls * patch_bank.expand(*target_shape) #B,-1,-1,-1,-1,-1) # B,nh,nw,K,ph,pw
        psel_sum = psel.sum(3).view(B*H*W,ph,pw,canvasd) # sum along K dimension  
    return psel_sum 


def diff_round(mu_x):
    mu_x_int = torch.round(mu_x).float()
    mu_x_diff = mu_x_int - mu_x 
    mu_x = mu_x + mu_x_diff.detach() # make it to closest int, while support gradient backprop 
    return mu_x 

def compute_filterbank_matrices(g_x, g_y, H, W, patch_size, grid, var=0.001):
    """ DRAW section 3.2 -- computes the parameters for an NxN grid of Gaussian filters over the input image.
    Args
        g_x, g_y -- tensors of shape (B, 1); unnormalized center coords for the attention window, suppose to 
                    be in range [-1,1]; but the DRAW model does not enforce that 
        logvar -- tensor of shape (B, 1); log variance for the Gaussian filters (filterbank matrices) on the attention window
        logdelta -- tensor of shape (B, 1); unnormalized stride for the spacing of the filters in the attention window
        H, W -- scalars; original image dimensions
        attn_window_size -- scalar; size of the attention window (specified by the read_size / write_size input args
    Returns
        g_x, g_y -- tensors of shape (B, 1); normalized center coords of the attention window;
        delta -- tensor of shape (B, 1); stride for the spacing of the filters in the attention window
        mu_x, mu_y -- tensors of shape (B, attn_window_size); means location of the filters at row and column
        F_x, F_y -- tensors of shape (B, N, W) and (B, N, H) where N=attention_window_size; filterbank matrices
    """ 
    B = g_x.shape[0]
    ph = patch_size 
    device = g_x.device

    # rescale attention window center coords and stride to ensure the initial patch covers the whole input image
    # eq 22 - 24
    delta = 1 # (B, 1)

    # compute the means of the filter
    # eq 19 - 20 [1,2,....10] - 5 = [-4,-3,....5]  
    offset_sampled_loc = torch.arange(1.0, 1.0+ph).to(device) - 0.5*ph
    ## offset_sampled_loc = torch.arange(1.0, 1.0+ph).to(device) - 0.5*ph
    offset_sampled_loc = offset_sampled_loc.view(1,ph).expand(B,-1)  # B,ph
    g_x = g_x.view(B,1).expand(-1,ph) # B,ph 
    g_y = g_y.view(B,1).expand(-1,ph) # B,ph 

    mu_x = g_x + offset_sampled_loc  # B,ph
    mu_x = diff_round(mu_x) 
    
    # g_y shape: B; + size (1,10) -> B,10
    mu_y = g_y + offset_sampled_loc # [B,ph]
    mu_y = diff_round(mu_y)

    # mu_x = g_x + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)
    # mu_y = g_y + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)
    # compute the filterbank matrices
    # B = batch dim; N = attn window size; H = original heigh; W = original width
    # eq 25 -- combines logvar=(B, 1, 1) * ( range=(B, 1, W) - mu=(B, N, 1) ) = out (B, N, W); then normalizes over W dimension;
    grid = grid.view(1,1,H).expand(B,ph,-1) # B,N,28

    mu_x = mu_x.view(B,ph,1).expand(-1,-1,H) # B,N,H 
    mu_y = mu_y.view(B,ph,1).expand(-1,-1,H) # B,N,H 

    F_x = torch.exp(-0.5/var * (grid - mu_x)**2) #B.N.H 
    # eq 26
    # F_x shape: B,1,28; mu_y shape: B,10,28 

    F_y = torch.exp(-0.5/var * (grid - mu_y)**2)
    # since Gaussian Filter Bank: require the intergration of F_y over its space is 1 
    return g_x, g_y, F_x, F_y, mu_x, mu_y #, grid 


#def zcls_BHWK_to_psel_BHW_phw(z_cls, info_dict, patch_bank): #, z_cls):
#    ''' obtain selected patches 
#    Args: 
#        z_cls (tensor): B,H,W,K output of ? 
#    Returns:
#        psel (tensor): B,H,W,K,ph,pw,canvasd -> BHW,ph,pw,canvasd
#    '''
#    B = z_cls.shape[0]
#    categorical_dim, nloci, nlocj, ph, pw, canvasd = info_dict['categorical_dim'], info_dict['nloci'], \
#            info_dict['nlocj'], info_dict['ph'], info_dict['pw'], info_dict['canvasd'] 
#    K,H,W = categorical_dim, nloci, nlocj 
#    CHECKSIZE(z_cls, (B,H,W,K))
#    # logger.info('check selectk: {}', z_cls.mean(3).view(-1)) 
#    CHECKSIZE(patch_bank, (1,H,W,K,ph,pw,canvasd))
#    target_shape = [B,H,W,K,ph,pw,canvasd]
#    z_cls = z_cls.view(B,H,W,K,1,1,1).expand(*target_shape) #-1,-1,-1,-1,ph,pw,canvasd) # B,nH,nW,K,ph,pw 
#    psel = z_cls * patch_bank.expand(*target_shape) #B,-1,-1,-1,-1,-1) # B,nh,nw,K,ph,pw
#    psel_sum = psel.sum(3).view(B*H*W,ph,pw,canvasd) # sum along K dimension  
#    return psel_sum 

def coords_ij_to_distmap(g_i, g_j,info_dict, grid_H,customize_nloc=0):
    '''  
    Args: 
        g_i (tensor): [B,Nloc], range(0,canvas_size) 
    Returns: 
        loc (tensor): [B*Nloc,canvas_size,canvas_size] 
    convert location i,j to a heatmap with peak at i,j 
    and radius as the patch-size 
    '''
    B = g_i.shape[0]
    csize = info_dict['canvas_size'] 
    categorical_dim, nloci, nlocj, ph =  info_dict['categorical_dim'], info_dict['nloci'], \
            info_dict['nlocj'], info_dict['ph'] 
    if customize_nloc:
        nloci=customize_nloc 
        nlocj=1
    K,H,W = categorical_dim, nloci, nlocj 
    CHECKSIZE(g_i, (B,H*W))
    cover = data_helper.get_cover(ph, csize) 
    var_img = ((cover // 2)/3.0) ** 2
    Nloc = H*W
    grid = grid_H.view(1,1,csize).expand(B*Nloc, 1, -1) # BN,1,28
    # B,1,28 - B,Nloc,1
    loc_x = torch.exp(- 0.5 / var_img * (grid - g_j.view(B*Nloc,1,1))**2) # BHW,   1, 28
    loc_y = torch.exp(- 0.5 / var_img * (grid - g_i.view(B*Nloc,1,1))**2) # B*NLoc,1, 28 
    CHECKSIZE(loc_x, (B*Nloc,1,csize))
    loc = loc_y.transpose(-2,-1) @ loc_x # expect: BHW,28,28  
    CHECKSIZE(loc, (B*Nloc,csize,csize))
    g_i_mask = g_i.ne(-1).view(B,Nloc,1,1).expand(-1,-1,csize,csize).view(
            B*Nloc,csize,csize).float() 
    loc = loc * g_i_mask # mask out those g_i = -1 
    return loc 

def transform_ij(g_i, g_j, psel, info_dict, grid_H, return_writted_loc=False):
    ''' transform patches to canvas 
    grid_H   = torch.arange(1., 1.+self.canvas_size).cuda().detach()
    Args: 
        g_i (tensor):  BNloc,1; the (center?) location of the selected patch 
        psel (tensor): BNloc,ph,pw,canvasd; the selected patches 
    Returns: 
        w (tensor):    BNloc,canvas_size,canvas_size,canvasd
        if return_writted_loc: 
            writed_loc: same shape as w, for the location being writed, entry is 1 

    H,W: number of location, put it in the batch dim, 
    if canvasd > 1: will also be intergrated into batch-dim
    '''
    BNloc = g_j.shape[0] 
    CHECKSIZE(g_j, (BNloc,1))

    canvas_size = info_dict['canvas_size'] 
    ph = info_dict['ph'] 
    canvasd = info_dict['canvasd']
    CHECK4D(psel) # last dim is the canvasd 
    canvasd = psel.shape[-1] 
    CHECKSIZE(psel, (BNloc,ph,ph,canvasd))
    ## K,H,W = categorical_dim, nloci, nlocj 
    # given the center location of the patches, and canvas_size, patch size 
    # F_x: shape (BNloc,10,28)
    g0_x, g0_y, F_x, F_y, mu_x, mu_y = compute_filterbank_matrices(
         g_j, g_i, canvas_size, canvas_size, ph, grid_H)
    CHECKSIZE(F_x, (BNloc,ph,canvas_size)) # BNloc,patch_size,canvas_size
    if canvasd > 1: # make if C,BNloc,1
        F_y = F_y.unsqueeze(0).expand(canvasd,-1,-1,-1).reshape(canvasd*BNloc,ph,canvas_size)
        F_x = F_x.unsqueeze(0).expand(canvasd,-1,-1,-1).reshape(canvasd*BNloc,ph,canvas_size)
    psel = psel.permute(3,0,1,2).contiguous().view(canvasd*BNloc,ph,ph) # 
    F_y_t = F_y.transpose(-2, -1) # BNloc,1
    w = F_y_t @ psel @ F_x # B*Nloc, canvas_size, canvas_size 
    w = w.view(canvasd,BNloc,canvas_size,canvas_size).permute(1,2,3,0).contiguous() 
    w = w.view(BNloc,canvas_size,canvas_size,canvasd)
    if return_writted_loc:
        writed_loc = F_y_t @ F_x # B*Nloc, canvas_size, canvas_size 

        writed_loc = writed_loc.view(canvasd,BNloc,canvas_size,canvas_size).permute(1,2,3,0).contiguous() 
        writed_loc = writed_loc.view(BNloc,canvas_size,canvas_size,canvasd)
        return w, writed_loc 
    return w 


