import math 
import numpy as np
from loguru import logger 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import utils 
from utils.checker import *
from utils import data_helper 
DEBUG=0 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # len, d_model 
        self.pe = pe

    def forward(self, x):
        # x: B.len.dim 
        CHECK3D(x)
        length = x.shape[1]
        B = x.shape[0]
        pe = self.pe[:length].view(1,length,self.d_model).to(x.device)
        return x + pe.expand(B,-1,-1) 

def BlockBN(in_c,out_c,k,s=1,p=0,BN=1):
    layers=[ nn.Conv2d(in_c, out_c, k, s, p)]
    if BN: 
        layers.append( nn.BatchNorm2d(out_c) )
    layers.append( nn.ReLU())
    return layers


def sample_logistic(mu, s, eps=1e-5):
    s = s.exp()
    U = torch.zeros_like(mu).uniform_(eps, 1-eps)
    return mu + s * torch.log( U / (1-U) )

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
    """ soft sampling """
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, latent_dim=None, categorical_dim=None, hard=False):
    """ warpper of soft sampling, support both soft and hard
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    if categorical_dim is not None:
        assert(logits.shape[-1] == categorical_dim)
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    
    if not hard:
        return y 
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 

class flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Permute(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'Permute{self.shape}'
    def forward(self, input):
        out = input.permute(*self.shape).contiguous() # shape)
        assert(out.is_contiguous())
        return out

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        ''' Reshapes the input according to the shape saved in the view data structure.  '''
        ## logger.info('input: {}, output: {}', input.shape, self.shape)
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class Scale(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'Scale into {self.shape}'
    def forward(self, input):
        ''' Reshapes the input according to the shape saved in the view data structure.  '''
        batch_size = input.size(0) 
        assert(len(input.shape) == 4), f'require 4D input; get {input.shape}'
        shape = (batch_size, *self.shape)
        out = F.interpolate(input, self.shape, mode='bilinear', align_corners=True)
        return out

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
    def __repr__(self):
        return f'Interpolate({self.size})'
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

#def concrete_rv_2_discrete(y, onehot=False): 
#    ''' y is output from gumbel_softmax, but reshape needed
#    Args: 
#        y = [..., categorical_dim] 
#    Returns: 
#        y_hard 1-of-K hot vector, [..., categorical_dim]
#    '''
#    shape = y.size()
#    _, ind = y.max(dim=-1)
#    if not onehot: return ind  
#    y_hard = torch.zeros_like(y).view(-1, shape[-1])
#    y_hard.scatter_(1, ind.view(-1, 1), 1)
#    y_hard = y_hard.view(*shape)
#    # Set gradients w.r.t. y_hard gradients w.r.t. y
#    y_hard = (y_hard - y).detach() + y
#    return y_hard  
#def to_scalar(arr):
#    if type(arr) == list:
#        return [x.item() for x in arr]
#    else:
#        return arr.item()


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


def get_conv_encoder(channels, latent_dim, embedding_dim, input_channel=3):
    out = [
            nn.Conv2d(input_channel, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_dim * embedding_dim, 1)
        ]
    return out 
def get_prepost_pross(img_size, out_dist_npara, imgd, pix_class, dec_hid, cfg, out_dist): 
    logger.debug('get img_size: {}', img_size) 
    postproc = []
    preproc = [] 
    canvas_size = data_helper.get_canvsize(cfg.dataset) 
    canvas_dim = data_helper.get_canvasd(cfg)
    #if cfg.multi_scale: 
    #    pass
    if img_size == 64 and canvas_size == 32: 
        logger.debug('out_dist_npara={}, imgd={}, pix_class={}', out_dist_npara, imgd, pix_class)
        postproc = [ nn.ConvTranspose2d(dec_hid, dec_hid, 2, stride=2),
            nn.BatchNorm2d(dec_hid), nn.ReLU(),
            nn.Conv2d(dec_hid,dec_hid,1,1), Scale((img_size, img_size)) ]
    elif img_size in [64, 32, 28]:
        preproc = [flatten()]
    else:
        raise NotImplementedError('Not Support: img_size={}, canvas_size={}'.format(img_size, canvas_size))
    # ---------------------------
    # deal with the output dist |
    # ---------------------------
    if out_dist in ['logistic', 'mixlogistic']: # , 'gaussian']: 
        outd = imgd*pix_class*out_dist_npara
        postproc.extend([nn.Conv2d(dec_hid, imgd*pix_class*out_dist_npara, 1, 1, bias=True)])
        postproc.extend([View((imgd, pix_class, out_dist_npara, img_size**2)), 
            # 0:B,1:d,2:cls,3:npara,4:H*W
            Permute((0, 1, 4, 2, 3)), # B, d, H*W,cls, npara
            View((imgd,img_size,img_size,pix_class,out_dist_npara)) 
            ])
    elif imgd == 1 and out_dist == 'bernoulli':
        oud = 1
        assert(imgd == 1), 'expect input to be [1,h,w] with 1 class'  
        assert(pix_class == 1 and out_dist_npara == 1), f'{pix_class}, {out_dist_npara}'
        postproc.append(nn.Conv2d(dec_hid, 1, 1, 1, bias=True))
        postproc.append(nn.Sigmoid())
    elif out_dist in ['mixture_logistic']: 
        assert(imgd==3), 'only support rgb images, and output dim is 10 dim '
        # if mixture_logistic: B,(imgd*3)*nmix+nmix; 
        # for each mixture: 1 for prob, 3(mean.var,gamma)x3(r,g,b)
        # imgd*pix_class*out_dist_npara, 
        outd = 100 
        postproc.extend([nn.Conv2d(dec_hid, outd, 1, 1, bias=False)])
    else:
        raise NotImplementedError 
    logger.debug('postproc: {}', postproc)
    logger.debug('preproc: {}', preproc)

    return preproc, postproc 

class MHeadout(nn.Module):
    def __init__(self, output_dims, input_dim, blockfun):
        super().__init__()
        self.layers = nn.ModuleList()
        for hid in output_dims:
            self.layers.append(nn.Sequential(
                *blockfun(input_dim, input_dim),
                nn.Linear(input_dim, hid)
                ))
        self.output_dim = sum(output_dims)
    def forward(self, input):
        out = []
        B = input.shape[0]
        CHECK2D(input)
        for layer in self.layers:
            out.append(layer(input))
        out = torch.cat(out, dim=1).view(B,self.output_dim)
        return out 

class MHeadin(nn.Module):
    ''' Multi-head embedding '''
    def __init__(self, input_dims, output_dim, blockfun):
        super().__init__()
        self.layers = nn.ModuleList()
        hid = int(output_dim // 2) 
        logger.info('[Init Headin] output_dim={}, input_dims={}', output_dim, input_dims)
        for input_dim in input_dims:
            self.layers.append(nn.Sequential(*blockfun(input_dim, hid))) ##nn.Linear(input_dim, hid)) 
        self.out_layer = nn.Sequential(
                *blockfun(hid*len(input_dims), hid),
                *blockfun(hid, hid),
                nn.Linear(hid, output_dim)
                )
        self.input_dims = input_dims
        self.hid_sum = hid * len(input_dims)

    def forward(self, input):
        '''
        input: (B,h1+h2+h3)
        '''
        out = []
        B = input.shape[0]
        sid = 0
        for layer, idim in zip(self.layers, self.input_dims):
            eid = idim + sid
            out.append(layer(input[:,sid:eid]))
            sid = eid 
        out = torch.cat(out, dim=1).view(B,self.hid_sum)
        out = self.out_layer(out)
        return out 
# -----------------------------------------------------------------------
#   get loc&sel sampler                                                 |
#   used for prior: predict sel+loc+stp, with cat,cat,ber distribution  |
# -----------------------------------------------------------------------

class prior_sampler(nn.Module):
    def  __init__(self, n_class_sel=None, n_class_loc=None, deterministic=False): 
        super().__init__()
        self.deterministic = deterministic
        assert(not self.deterministic)
        self.n_class_sel = n_class_sel 
        self.n_class_loc = n_class_loc 
        self.loc_mem = None

    def clean_mem(self):
        self.loc_mem = None # B,K, binary, if 1: this entry is selected before;
    def convert_sample_out2dict(self, sampled_out):
        """convert the sampled output into dict 
        Args: 
            sampled_out: B,nlen,D
        return torch.cat([sel_onehot, loc_onehot, logits_stp], dim=1)
        """
        CHECK3D(sampled_out)
        B,l,D = sampled_out.shape
        sampled_out = sampled_out.view(B,l,self.n_class_sel+self.n_class_loc+1)
        sel_onehot = sampled_out[:,:,:self.n_class_sel]
        loc_onehot = sampled_out[:,:,self.n_class_sel:self.n_class_sel+self.n_class_loc]
        stp_output = sampled_out[:,:,-1:] # N,D
        return {'sampled_sel': sel_onehot, 'sampled_loc': loc_onehot, 'sampled_stp': stp_output}

    def forward(self, out, mask_out_prevloc=False, is_generate=False):
        ''' 
        Sampled the 1-of-k hot vector 
        Args: 
            out: B*Len,nD, p1 [n_class_sel] for selection, 
                p2 [n_class_loc] for location, 
                p3 [1] for stp prediction 
            if mask_out_prevloc is True: 
                the logits at loc selected before will be assigned with lowest score
                only supports during sampling 
        '''
        n_class_sel, n_class_loc = self.n_class_sel, self.n_class_loc
        CHECK2D(out) 
        B,_ = out.shape
        CHECKSIZE(out, (B,n_class_sel+n_class_loc+1)) 
        # -- patch selection -- 
        logits_loc = out[:,:n_class_sel] # B,K
        p_y = torch.softmax(logits_loc, -1)
        y_sample_ = torch.multinomial(p_y, 1) # (B,1)
        #if self.deterministic: 
        #    y_sample_ = p_y.max(1)[1]
        y_sample = F.one_hot(y_sample_, n_class_sel).float() 
        sel_onehot = y_sample.view(B,n_class_sel)

        # -- location -- 
        logits_loc = out[:,n_class_sel:n_class_sel+n_class_loc]
        p_y = torch.softmax(logits_loc, -1)
        if is_generate and mask_out_prevloc and self.loc_mem is not None: # keep the value of zeros entry in self.loc_mem 
            # raise NotImplementedError('not checked ')
            # for the entry with self.loc_mem[i] == 1: gives 0
            p_y = p_y * (1-self.loc_mem) # B,1
        y_sample_loc = torch.multinomial(p_y, 1) # (B,1)
        #if self.deterministic:
        #    y_sample_loc = p_y.max(1)[1]
        y_sample = F.one_hot(y_sample_loc, n_class_loc).float() 
        loc_onehot   = y_sample.view(B,-1)
       
        # -- stop selction -- 
        logits_stp = torch.bernoulli(out[:,-1]).float().view(B,1)
        #if self.deterministic:
        #    logits_stp = (out[:, -1] > 0.5).float().view(B,1)
        # -- for stp=0, deactivated the sel and loc, set one-hot as all zero -- 
        sel_onehot = sel_onehot * logits_stp
        loc_onehot = loc_onehot * logits_stp
        return torch.cat([sel_onehot, loc_onehot, logits_stp], dim=1)

