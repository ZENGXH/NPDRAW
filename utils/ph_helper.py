''' p_h(z|x) related helper functions '''
import os, pickle, json 
import sys 
import numpy as np
from loguru import logger
import time 
import torch
from torch import nn, optim
from torch.nn import functional as F
from utils import model_helper, data_helper
from utils.checker import *
# from utils.io_helper import read_strk_info_mnist 
import utils
from model.modules import prior_sampler
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib import cm 
LESS_VERBOSE = int(os.environ.get('LESS_VERBOSE', 0))
# -------------------------
#  location transform     |
# -------------------------
def loc_transform_i2f(float_matrix, loc_index): 
    ''' differentiable
    Args: 
        float_matrix: 1,N,2 
        loc_index: (B,N), one-hot vector 
    Returns: 
        loc_ij (B,2), float location 
    '''
    float_matrix = float_matrix.to(loc_index.device).detach()
    CHECK2D(loc_index)
    B = loc_index.shape[0]
    loc_ij = (float_matrix.expand(B,-1,-1) * loc_index.unsqueeze(2)).sum(1)
    return loc_ij 

def loc_transform_f2i(float_matrix, loc_ij): 
    ''' quantization, require grad? no
    expand both into B,N,2 compute l2 distance between B*N pair 
    and take the smallest dist
    Args: 
        float_matrix: 1,N,2 
        loc_ij (B,2), float location, value range: [0, canvas_size]
    Returns: 
        loc_index: (B,N), one-hot vector 
    '''
    float_matrix = float_matrix.to(loc_ij.device)
    CHECK2D(loc_ij)
    CHECK3D(float_matrix)
    N = float_matrix.shape[1] 
    B = loc_ij.shape[0] 
    loc_ijn = loc_ij.unsqueeze(1).expand(-1,N,-1) # (B,2), B=batch_size*nloc, N=num_class_of_loc
    dist = (loc_ijn - float_matrix.expand(B,-1,-1)).pow(2).sum(-1) # B,N 
    _, assign = dist.min(1) # min along dim1 (Ncat)
    loc_index = F.one_hot(assign, N).float() 
    return loc_index 


## -------------------------
##   build gaussian map    |
## -------------------------
#def get_gaussian_map(size, device=None):
#    ''' create a gaussian map with 3 sigma at R, size//2
#    https://github.com/microsoft/human-pose-estimation.pytorch/blob/c3a30c0e1f83e73b3038b1a443becf6b4a19cf1f/lib/dataset/JointsDataset.py#L187 
#    '''
#    x = np.arange(0, size, 1, np.float32) 
#    y = x[:, np.newaxis] 
#    x0 = y0 = size // 2 # center 
#    sigma = (size // 2) / 3.0 # sigma value 
#    # The gaussian is not normalized, we want the center value to equal 1 
#    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#    g = torch.from_numpy(g)
#    if device is not None:
#        g = g.to(device)
#    return g
#
def get_gt_sel_loc(info_dict, full_prior_gt, xid):
    """ load gt for canvas's patch sel and loc 
    # provided gt of patch sel or loc, during testing 
    # used for debugging / decomposing bce 
    Parameters
    ----------
        xid : tensor, [Number_sample]

    Returns 
    -------
        y_flat: one hot tensor (B,nloc,K)
        logits_loc: range in [0,1]; (B,nloc,2) 
            loc of the center  
    """
    K, N, canvas_size, ph = info_dict['categorical_dim'], \
        info_dict['latent_dim'], info_dict['canvas_size'], info_dict['pw']
    B_new = xid.shape[0]
    prior_gt = full_prior_gt.index_select(0, xid)
    size_g   = prior_gt.shape[-1]
    prior_gt = prior_gt.view(B_new,-1,size_g) 

    # B,nloc
    prior_gt_sel = prior_gt[:,:,0].long() # gt selction 
    # offset 1: reason: -1 means zero patches 
    prior_gt_sel = F.one_hot(1+prior_gt_sel.long(), K+1)[:,:,1:].float() # B,nloc,K 
    y_flat = prior_gt_sel

    # B,nloc,2; 
    prior_gt_loc = (prior_gt[:,:,1:3] * prior_gt[:,:,-1].unsqueeze(2)) \
            + ph*0.5 
    prior_gt_loc = prior_gt_loc / canvas_size # norm to [0,1] 
    logits_loc = prior_gt_loc.float()   
    return y_flat, logits_loc 

def get_n_class_loc(cfg):
    canvas_size = data_helper.get_canvsize(cfg.dataset)
    if cfg.vary_loc_vae.anchor_dependent:
        if not LESS_VERBOSE: logger.info('[get_n_class_loc]: anchor_dependent: canvas_size={}, stride={}, nloc={}', 
            canvas_size, cfg.vary_loc_vae.loc_stride, cfg.vary_loc_vae.nloc)
        float_matrix = prepare_loc_float_matrix_blockdpd(
            canvas_size, cfg.vary_loc_vae.loc_stride, cfg.vary_loc_vae.nloc)
    else:
        float_matrix = prepare_loc_float_matrix(
            canvas_size, cfg.vary_loc_vae.loc_stride)
    n_class_loc = float_matrix.shape[1]
    return n_class_loc

def prepare_gt(gt_file, Nloc, centered_patch_lt, add_stop=False, add_empty=False, cached_dir=None):
    ''' read the gt's loc and pid 
    Args:
        gt_file (list): len=#img -> entry: list, len=#steps -> entry: [pid, loci, locj]
        loci, locj is for the top-left point 
    Returns: 
        gt_array (ndarray): [Nsample,Nloc,5] 
            1: pid \in {-1, ..., K}
            2&3: location of the top left point of the patches 
            4: mask for pid; this mask is not valid for the bfs3 cases, 
                since its determined by the len of gt_file; not valid entry  
            5: mask for location 
    '''
    if cached_dir is not None: 
        cached_f = '%s_gt_%d-%d-%d-%d.npy'%(cached_dir, Nloc, centered_patch_lt, add_stop, add_empty)
        if os.path.exists(cached_f): 
            if not LESS_VERBOSE: logger.info('[load gt_file] directly from cached: {}', cached_f) 
            return torch.from_numpy( np.load(cached_f) ) 
    else:
        cached_f = None
    # gt_file = trim_empty_patch(gt_file, add_stop)
    if not LESS_VERBOSE: logger.info('load gt_file: #{}', len(gt_file))
    has_negative_pid = sum([sum([s[0] == -1 for s in v]) for k,v in gt_file.items()]) > 0
    def toarray(i):
        i = [np.array(j) for j in i]
        return np.array(i)
    nloc = []
    gt_array = np.zeros((len(gt_file), Nloc, 5)) - 1 #init with -1
    gt_array[:,:,3:] = 0 # set dim 3&4 as zero 
    gt_array[:,:,2] = centered_patch_lt # [0.center]

    for k in gt_file.keys():
        gt_curimg = toarray(gt_file[k]) #Nsteps,3
        imgid = int(k) 
        if not add_stop and gt_curimg.min() == -1: # has invalid entry 
            raise NotImplementedError # expected to be removed in trim_empty_patch
        if len(gt_curimg) >= Nloc:
            gt_array[imgid,:,:3] = gt_curimg[:Nloc, :3]
            gt_array[imgid,:,3:] = 1
            nloc.append(Nloc)
        else: # gt_curimg < Nloc; less gt then required  
            ngt = len(gt_curimg)
            if ngt <= 0: 
                if not LESS_VERBOSE: logger.info('get ngt: {}, k={}', ngt, k) 
                continue 
            gt_array[imgid,:ngt,:3] = gt_curimg[:,:3] 
            gt_array[imgid,:ngt,3:] = 1
            nloc.append(ngt)

        if add_stop or add_empty:
            for l in range(len(gt_curimg)): # Nsteps, 3 
                if gt_curimg[l,0] == -1: # cls id = -1
                    assert(gt_curimg[l,1] == -1 and gt_curimg[l,2] == -1), 'get {}'.format(gt_curimg[l])
                    gt_array[imgid,l,3:] = 0 # reset all the entry for this step, this image as 0 
    assert(not add_empty), 'not suport now'
    if add_empty:
        gt_array[np.where(gt_array[:,:,0] == -1), -1] = 0
        gt_array[:,:,0] = gt_array[:,:,0] + 1 # offset the index by 1
        if not LESS_VERBOSE: logger.info('[CHECK]: add_empty: number of empty patch: {}', (gt_array[:,:,0]==0).astype(np.uint8).sum())

    nloc = np.array(nloc)
    if not LESS_VERBOSE: logger.info('read all gt: {}; min nloc={}, max={}; mean={:.2f}',
        len(nloc), nloc.min(), nloc.max(), nloc.mean())
    if cached_f is not None: 
        np.save(cached_f, gt_array)
    return torch.from_numpy(gt_array)


def load_prior_with_patch_bank(info_dict, cfg, device, metric=None, n_class_loc=None):
    ''' expand patch_bank to be 1,nloc,nloc,K,patchH,patchW 
    freq_init to be 1,nH*nW==latent_dim,K
    Return: 
        prior_gt: [Nsample.nloc,5], 5 for [cls,top-left-i,top-left-j,??valid-cls,??valid-loc]
        patch_bank in shape [1,nH,nW,K,ph,pw,imgd]
    '''
    # create patch_bank
    canvas_size = info_dict['canvas_size']
    pw = cfg.ww 
    ph = cfg.wh 
    patch_size = pw 
    n_class_loc = get_n_class_loc(cfg) if n_class_loc is None else n_class_loc
    assert(cfg.cat_vae.canvas_dim)
    canvasd = cfg.cat_vae.canvas_dim
    assert(canvasd > 0), 'require canvasd > 0'
    centered_patch_lt = canvas_size//2 - pw//2 
    nloci = cfg.vary_loc_vae.nloc #5 
    nlocj = 1 #1 # 2*3 = 6 
    
    prior_weight = cfg.prior_weight 
    if not LESS_VERBOSE: logger.info('[load prior_weight] {}', prior_weight) 
    
    # ---- load args of prior model ----
    add_empty = 0
    inputd = 2 
    from prior_config import cfg as prior_cfg 
    output_folder = os.path.dirname(cfg.prior_weight)
    if os.path.exists(f'{output_folder}/cfg.yml'): 
        if not LESS_VERBOSE: logger.info('[load prior cfg] {}/cfg.yml', output_folder) 
        prior_cfg.merge_from_file(f'{output_folder}/cfg.yml')
        if not LESS_VERBOSE: logger.info('[p_h(z|x)] {}', prior_cfg.gt_file) 
        if metric is not None:
            metric.write('prior', 'https://www.comet.ml/zengxh/patch2img/'+prior_cfg.exp_key)
            metric.write('gt', prior_cfg.gt_file)
        if os.environ.get('EVAL') is not None: # write the collection file if needed 
            with open('.results/eval_out/%s.md'%(os.environ.get('EVAL')), 'a') as f:
                f.write('[prior] https://www.comet.ml/zengxh/patch2img/'+prior_cfg.exp_key+'\n')
                f.write('[gt] %s'%prior_cfg.gt_file+'\n')
    else:
        logger.warning('load cfg from args.pkl will be depreciated')
        with open(output_folder + '/args.pkl', 'rb') as f:
            loaded_prior_args = pickle.load(f)
        args_list = []
        for k,v in vars(loaded_prior_args).items():
            if k in ['comet', 'sampled_only', 'eval_only', 'model_path']: continue 
            args_list.append(str(k))
            args_list.append(str(v))
        prior_cfg.merge_from_list(args_list)
    logger.debug('cfg after merged: {}', prior_cfg)
    prior_modelp = os.path.dirname(cfg.prior_weight) 

    def has_cached(prior_modelp, prefix):
        for f in os.listdir(prior_modelp):
            if prefix in f: return 1
        return 0
    if has_cached(prior_modelp, 'cached_gt'):
        logger.info('load cache, skip reading full data')
        cached_dir = prior_modelp + '/cached_gt' 
        gt_file, gt_file_eval = {}, {} 
    elif os.path.exists('%s/gt.pkl'%prior_modelp):
        if not LESS_VERBOSE: logger.info('[load prior_gt] from {}/gt.pkl', prior_modelp)
        gt = pickle.load(open('%s/gt.pkl'%prior_modelp, 'rb'))
        cached_dir = prior_modelp + '/cached_gt' 
        gt_file = gt['patchid_train']
        gt_file_eval = gt['patchid_eval']
    else:
        if not LESS_VERBOSE: logger.info('[load prior_gt] {}', prior_cfg.gt_file) 
        gt = json.load(open(prior_cfg.gt_file, 'r'))
        cached_dir = os.path.dirname(prior_cfg.gt_file) + '/cached_gt' 
        gt_file = gt['patchid_train']
        gt_file_eval = gt['patchid_eval']
    # make sure the patch bank is consistent with prior model

    # ---- read patch bank ----
    pb_f = os.path.dirname(prior_cfg.gt_file) + '/patch_bank.npy' # prefer from gt path 
    pb_npy = '%s/patch_bank.npy'%prior_modelp  # second prefered 
    if not LESS_VERBOSE: logger.info('start load patch_bank: {}', pb_npy)
    if os.path.exists(pb_npy):
        patch_bank = np.load(pb_npy)
    elif os.path.exists(pb_f):
        patch_bank = np.load(pb_f)
    else:
        raise FileNotFoundError(pb_npy)
    #else:
    #    patch_bank = utils.helper.load_patch_bank(gt['patch_name'], gt['cluster_model_name'])
    #    np.save(pb_f, patch_bank)

    if patch_bank.max() > 1: 
        patch_bank = patch_bank / 255.0

    loaded_patch_bank = torch.from_numpy(patch_bank).float().to(device) # K, WW, WH 
    K = loaded_patch_bank.shape[0] 
    if not LESS_VERBOSE: logger.info('Build Prior Model, patch_bank: {} | expect K={}, pw={}, d={}',  
        loaded_patch_bank.shape, K, pw, canvasd)
    CHECKEQ(loaded_patch_bank.view(-1).shape[0], K*pw*ph*canvasd) 
    # ---- Init Model ----
    assert(prior_cfg.model_name == 'cnn_prior' and prior_cfg.use_vit)
    assert(not prior_cfg.concat_one_hot)
    #assert(not prior_cfg.input_id_canvas)
    #assert(not prior_cfg.input_id_only)
    
    kwargs = {}
    from model.vit_prior import CNNHead
    kwargs.update({'input_id_only': prior_cfg.input_id_only,
        'nloc': prior_cfg.nloc,
        'concat_one_hot': prior_cfg.concat_one_hot, 
        'use_cnn_process': prior_cfg.use_cnn_process,
        'mlp_dim': prior_cfg.vit_mlp_dim, 
        'vit_dropout': prior_cfg.vit_dropout,
        'depth': prior_cfg.vit_depth })
    hid_dim=prior_cfg.hidden_size_prior # 256
    if prior_cfg.mhead:
        head_dims = [K, n_class_loc, 1]
    else:
        head_dims = None
    # turn partial function into object class 
    fun_prior_sampler = prior_sampler(n_class_sel=K, n_class_loc=n_class_loc) 
    ## partial(prior_sampler, n_class_sel=K, n_class_loc=n_class_loc)
    loaded_prior_model = CNNHead(rnn_hid=hid_dim, 
        mask_out_prevloc_samples=prior_cfg.mask_out_prevloc_samples,
        canvasd=canvasd, canvas_size=canvas_size,
        input_dim=canvasd+1, sampler=fun_prior_sampler, 
        pos_encode=prior_cfg.nloc if prior_cfg.pos_encode else 0,
        use_emb_enc=prior_cfg.use_emb_enc,
        activation='sigmoid_last_dim',
        kernel_size=prior_cfg.kernel_size, head_dims=head_dims,
        num_layers=prior_cfg.num_layers,
        USE_FC_ENC=len(prior_cfg.start_time)==0 or \
                prior_cfg.start_time < '20-10-16-18-29-19',
                **kwargs
        ).to(device) 
    logger.debug('[build prior model] {}', loaded_prior_model)
    prior_state = torch.load(prior_weight) 
    if 'model' in prior_state:
        prior_state = prior_state['model']
    loaded_prior_model.load_state_dict(prior_state) # load prior model 
    model_helper.freeze_parameter(loaded_prior_model)

    # ---- parse gt info p_h(z|x) file ----
    if not LESS_VERBOSE: logger.info('parse GT with slide-based primitive')
    add_stop = cfg.cat_vae.add_stop 
    prior_gt_train = prepare_gt(gt_file, nloci*nlocj,     centered_patch_lt, add_stop=add_stop, cached_dir=cached_dir+'train').to(device) 
    prior_gt_eval = prepare_gt(gt_file_eval, nloci*nlocj, centered_patch_lt, add_stop=add_stop, cached_dir=cached_dir+'eval').to(device)  
    loaded_prior_gt = torch.cat([prior_gt_train, prior_gt_eval]) 
    # read_info_f = read_strk_info_mnist 

    #if cfg.prior_model.eval_with_freq:
    #    raise NotImplementedError 
    nH, nW = nloci, nlocj
    loaded_patch_bank = loaded_patch_bank.view(1,K*pw*ph*canvasd).expand(nH*nW, -1).view(
            1,nH,nW,K,ph,pw,canvasd) # 1,HW,K,ph*pw
    if not LESS_VERBOSE: logger.info('set patch_bank shape {}', loaded_patch_bank.shape)
    return loaded_patch_bank, loaded_prior_model, loaded_prior_gt

#def plot_float_matrix(float_matrix, canvas_size, output_path='float_matrix_vis.png', patch_size=2):
#    cml = cm.rainbow # from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
#    n_anchor, n_loc, _ = float_matrix.shape 
#    ww = wh = patch_size
#    frames_loc = [] 
#    fig, ax = plt.subplots(1,max(2,n_anchor),figsize=(25, 3), subplot_kw=dict(xticks=[], yticks=[]))
#    fig.subplots_adjust(hspace=0.001, wspace=0.05)
#    if not LESS_VERBOSE: logger.info('[POS INFOR] covered range for each anchor: {:.1f}', float_matrix[0,:,0].max() - float_matrix[0,:,0].min())
#    if not LESS_VERBOSE: logger.info('[POS INFOR] dist between two avail pos for one anchor: {:.1f}', float_matrix[0,1,1] - float_matrix[0,0,1])
#    if n_anchor > 1:
#        if not LESS_VERBOSE: logger.info('[POS INFOR] cloest distance for box at neighbor anchor {:.1f}', float_matrix[1,:,0].min() - float_matrix[0,:,0].max())
#
#    for ida in range(n_anchor):
#        img_center = np.zeros((canvas_size, canvas_size,3)).astype(np.uint8)
#        img_center = Image.fromarray(img_center)
#        draw = ImageDraw.Draw(img_center)
#        for idl in range(n_loc):
#            ti, tj = float_matrix[ida, idl] - 0.5*patch_size
#            # draw the center 
#            ri_color = tuple([int(c*255) for c in list(
#                cm.rainbow((ida*25+idl)%cml.N))[:3]])
#            draw.rectangle([(int(tj), int(ti)), (int(tj+wh), int(ti+ww))] , outline=ri_color)
#        helper.myimshow(ax.flat[ida], img_center)
#        ## frames_loc.append(img_center)
#    plt.tight_layout() 
#    fig.savefig(output_path)
#    return output_path
#
