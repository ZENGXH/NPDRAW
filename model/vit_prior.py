import time
import math 
import numpy as np
from loguru import logger 
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils 
from utils.checker import *
from .modules import MHeadin, MHeadout,flatten
from .vit_pytorch import ViT  

class CNNHead(nn.Module):
    name='cnn_head'
    ''' CNN_Prior: generate sel_t ~ Cat(pi; canvas_{t-1}, loc_{t-1}), loc_t ~ Cat(), stp_t ~ Ber()
    Args: 
        input_dim: in our case, it is the number of classes 
        other_d: in out case, its the location parameters 
    '''
    def __init__(self, rnn_hid, input_dim, sampler, pos_encode, use_emb_enc, 
            activation, 
            mask_out_prevloc_samples, #  required 
            use_cnn_process, concat_one_hot, input_id_only,
            use_fast_transformers=0,
            canvasd=1, mlp_dim=-1, depth=-1,vit_dropout=1,
            canvas_size=-1,
            nloc=None,
            head_dims=None, canvas_plotter=None, geometric=0, 
            # not used 
            canv_init=0,
            cond_on_loc=False, **kwargs):  
        super().__init__()
        use_picnn = False  
        # self.canv_init = canv_init
        self.use_fast_transformers = use_fast_transformers
        self.canvas_plotter = canvas_plotter
        self.pos_encode = pos_encode
        self.geometric = geometric
        # self.mask_out_prevloc_samples = mask_out_prevloc_samples # if True; not allow the model repeat location during sampling 
        self.sampler = sampler # prior_sampler defined in model.modules  
        self.input_dim = input_dim ## self.canvas_plotter.imgd ## input_dim  
        # not a good way: 
        self.imgd = canvasd ## self.canvas_plotter.imgd
        self.canvasd = canvasd
        self.canvas_size = canvas_size
        self.nloc = nloc
        # 
        self.activation = activation
        self.rnn_hid = rnn_hid
        self.cat_one_hot_dim = sum(head_dims)
        assert(rnn_hid >= -1) # not support -2 anymore, use mhead=1 instead 

        if cond_on_loc: assert(head_dims), 'require head_dims'
        def _Block(in_c,out_c):
            layers = [ nn.Linear(in_c, out_c), nn.ReLU() ] 
            return layers
        blockfun_nbn = _Block

        self.use_cnn_process = use_cnn_process 
        self.concat_one_hot = concat_one_hot
        self.input_id_only = input_id_only 
        assert(not self.input_id_only), 'prevent wrong configuration'
        ## if self.concat_one_hot: raise NotImplementedError 
        #assert(self.use_cnn_process) 
        ## and self.concat_one_hot), '%s, %s'%(self.use_cnn_process, self.concat_one_hot)
        ## mlp_dim = 1024
        if self.use_fast_transformers:
            self.nheads = 8 
        else:
            if mlp_dim <= 56:
                self.nheads = 4
            elif mlp_dim <= 128:
                self.nheads = 8
            else:
                self.nheads = 16

        logger.info('[VIT Prior] use_cnn_process: {}, concat_one_hot:'
            ' {}, input_id_only: {}, canvas_size: {}, canvasd: {}, '
            'nloc={}, mlp_dim={}, nhead={}', self.use_cnn_process, 
            self.concat_one_hot, self.input_id_only, self.canvas_size, 
            canvasd, nloc, mlp_dim, self.nheads)
        if self.use_cnn_process:
            self.input_csize = 13 ## 144 ## self.canvas_size // 2 # 14 
            self.vit_channel = 1
            # 28 -> 14 -> 6 
            # 32 -> 16 -> 7 
            self.preprocess = nn.Sequential(
                    nn.Conv2d(self.input_dim, 16, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(),
                    nn.Conv2d(16, 16, 3, 2),    
                    flatten(),
                    nn.Linear(16*(self.canvas_size//4-1)**2, 
                        self.vit_channel*(self.input_csize**2))                    
                    ) ## 16-16 x 14x14 
            #with torch.no_grad():
            #    x = torch.zeros(1,self.input_dim,self.canvas_size,self.canvas_size)
            #    x_out = self.preprocess(x)
            #    print(x_out.shape)
            #    print(x.shape)
            if mlp_dim == 2048:
                self.preprocess = nn.Sequential(
                        nn.Conv2d(self.input_dim, 16, 3, 2, 1), nn.ReLU(),
                        nn.Conv2d(16, 16, 1, 1 ), 
                        flatten(),
                        ) ## 16-16 x 14x14 
                self.vit_channel = 16 
                self.input_csize = self.canvas_size//2 #14 
            assert(not self.concat_one_hot)
            #if self.concat_one_hot:
            #    DHW = self.vit_channel*(self.input_csize**2)
            #    #DHW = self.input_dim*self.canvas_size**2
            #    #self.input_csize = 128
            #    #self.vit_channel = 1
            #    self.enc_mlp = MHeadin( [1,DHW],
            #            # head_dims.append(DHW), 
            #            self.vit_channel*(self.input_csize**2),
            #            _Block)
            #    self.input_emd_dim = self.vit_channel*(self.input_csize**2)
        else:
            self.vit_channel = self.input_dim
            self.input_csize = self.canvas_size
        logger.info('[Args] dim={}, mlp_dim={}, channels={}, heads={} input_csize={}, nloc={}',
            self.rnn_hid, mlp_dim, self.vit_channel,self.nheads, self.input_csize, self.nloc)
        if self.use_fast_transformers:
            from .fast_transformer import FastTransform
            self.enc = FastTransform( 
                image_size=self.input_csize * self.nloc,
                patch_size=self.input_csize, 
                max_length=self.nloc,
                dim=self.rnn_hid, depth=depth, 
                mlp_dim=mlp_dim, ## 2048, 
                channels=self.vit_channel, 
                heads=self.nheads, 
                dropout=0.1 if vit_dropout else 0, 
                emb_dropout=0.1 if vit_dropout else 0
                )
        else: 
            self.enc = ViT( 
                image_size=self.input_csize * self.nloc,
                patch_size=self.input_csize, 
                dim=self.rnn_hid, depth=depth, 
                mlp_dim=mlp_dim, ## 2048, 
                channels=self.vit_channel, 
                heads=self.nheads, 
                dropout=0.1 if vit_dropout else 0, 
                emb_dropout=0.1 if vit_dropout else 0
                )

        # map one-hot vector into a feat 
        logger.info('[Init Prior Model] head_dims: {}', head_dims)

        if self.pos_encode and not use_picnn:
            raise NotImplementedError('require with use_picnn')
        if self.pos_encode:
            self.generate_loc_feat = PositionalEncoding(rnn_hid, max_len=self.pos_encode) 
        assert(head_dims is not None and not cond_on_loc) 
        #if head_dims is None: ## or not use_picnn: 
        #    self.compute_q_dist = nn.Sequential(
        #        *_Block(rnn_hid, rnn_hid),
        #        nn.Linear(rnn_hid, input_dim)
        #        )
        #elif cond_on_loc: # head_dims: pid, loc, stp 
        #    self.compute_q_dist = MHeadoutCond(head_dims, rnn_hid, 1)
        #else:
        #    self.compute_q_dist = MHeadout(head_dims, rnn_hid, _Block)
        self.compute_q_dist = MHeadout(head_dims, rnn_hid, _Block)
        self.head_dims = head_dims
        assert(activation == 'sigmoid_last_dim')

    def evaluate(self, x): 
        """ compite KL 
        Args: 
            x: B,nloc,ncls_sel+ncls_loc+1
        """
        assert(self.canvas_plotter is not None)
        CHECK3D(x)
        B,nloc,cat_one_hot_dim = x.shape
        CHECKEQ(cat_one_hot_dim, self.cat_one_hot_dim)
        inputd = self.canvasd + 1
        # create canvas input | the follow process is copied from runner/cnn_prior1.py 

        # -- create input canvas from gt --
        H = W = csize = self.canvas_size
        gts = inputs_one_hot = x # B,nloc,cat_one_hot_dim
        td = {}
        tic = time.time()
        sample_dict = self.sampler.convert_sample_out2dict(gts) 
        
        td['convert'] = '%.4f'%(time.time() - tic)
        tic = time.time()

        gt_canvas, loc_map = self.canvas_plotter.create_canvas(sample_dict, break_in_step=1, 
                return_per_step_loc=0) 
        td['plot'] = '%.4f'%(time.time() - tic) 
        tic = time.time()

        gt_canvas = gt_canvas.view(B,nloc,self.imgd,csize,csize) # B,nloc,imgd,Hc,Wc 
        loc_map = loc_map.view(B,nloc,1,csize,csize) # B,nloc,imgd,Hc,Wc 
        # concat input with loc map 
        gt_canvas = torch.cat([gt_canvas, loc_map], dim=2) # B,nloc,imgd+1,csize,csize
        
        gt_canvas  = torch.cat([gt_canvas.new_zeros(B,1,inputd,csize,csize), 
            gt_canvas[:,:-1]], dim=1)
        # the last gt_canvas is the final image, which dont have 'next step' gt; 
        gt_canvas = gt_canvas.view(B*nloc,inputd,csize,csize)
        # output, sampled_out = self.prior(gt_canvas)

        output, sampled_out = self.forward(gt_canvas, inputs_one_hot.view(B*nloc,-1)) 

        td['fw'] = '%.4f'%(time.time() - tic) 
        tic = time.time()
        # logger.info('time: {}', td)
        output = output.view(B,nloc,self.cat_one_hot_dim)
        sampled_out = sampled_out.view(B,nloc,self.cat_one_hot_dim)
        return output, sampled_out


    def forward(self, input, one_hot_prev, mask_out_prevloc=False, is_generate=False, nloc=None): 
        """ 
        Args: 
            x shape: B,Len,Ndim  
                Len can be specified through nloc, usually during generation, otherwise will assume the 
                length is default value 
            (only passed when sampling, otherwise bydefault false): mask_out_prevloc 
        Returns: 
            p(z_t|z_{<t}): gives the likelihood at each step (for categorical variable, 
                the likelihood is not normalized, for bernoulli, its normalized)
            z_t ~ p(z_t|z_{<t}): can be used to visualize the p(z_t|z_{<t})

        out&input: channel Ndim: 
            p1 [n_class_sel] for selection, 
            p2 [n_class_loc] for location, 
            p3 [1] for stp prediction 
        """ 
        if is_generate: assert(nloc), 'nloc args require '
        N = self.nloc if nloc is None else nloc # turn in to sequence 
        BN,D,H,W = input.shape 
        HW = H*W
        CHECKSIZE(input, (BN,self.input_dim,self.canvas_size,self.canvas_size)) 
        B = BN // N 
        C = sum(self.head_dims) 
        input = input.view(B,N,D*HW)  # view as B,Npat,D*H*W 
        CHECKSIZE(one_hot_prev, (BN,self.cat_one_hot_dim)) 

        # -- forward Transformer -- 
        ## if not is_generate: # training mode: create mask to prevent cheating 
        mask = torch.tril(torch.ones((N+1,N+1))).to(input.device) 
        mask = mask.view(1,1,N+1,N+1).expand(B,self.nheads,-1,-1).bool()
        if self.input_id_only:
            id_emb = self.enc_mlp(one_hot_prev).view(B,N,D*HW) 
            input_emd  = self.enc(id_emb, mask) ## .view(B,self.rnn_hid,-1).sum(-1)
        else:
            if self.use_cnn_process:
                #logger.info('input: {} | input_emd_dim: {} B={},N={}', input.shape, self.input_emd_dim, B, N)
                input = self.preprocess(input.view(
                    B*N,self.input_dim,self.canvas_size,self.canvas_size)).view(B,N,-1)
                # logger.info('input: {} | input_emd_dim: {}', input.shape, self.input_emd_dim)
                if self.concat_one_hot: 
                    cat_input = torch.cat( 
                        [   one_hot_prev.view(B*N,self.cat_one_hot_dim)[:,-1:],  
                            input.view(B*N,self.input_emd_dim) 
                        ], dim=1)
                    # logger.info('cat_input: {} {}; {}', cat_input.shape, one_hot_prev.shape, self.enc_mlp)
                    input = self.enc_mlp(cat_input).view(B,N,-1) ## input = torch.cat([input, id_emb])
            input_emd = self.enc(input, mask)  
        CHECKSIZE( input_emd, (B,N,self.rnn_hid) ) # output is vector 

        x = input_emd 
        h_1d = x.reshape(B*N,self.rnn_hid)  # B,N,C
        q_dist_raw = self.compute_q_dist(h_1d) # B,N,C'
        q_dist_raw = q_dist_raw.view(B*N,-1)
        if self.activation == 'sigmoid_last_dim': 
            q_dist_last_dim = q_dist_raw[:,-1].sigmoid().unsqueeze(1)
        q_dist = torch.cat([q_dist_raw[:,:-1], q_dist_last_dim], dim=1) 
        # -- draw samples -- 
        sampled_out = self.sampler(q_dist, mask_out_prevloc=mask_out_prevloc, is_generate=is_generate).view(B,N,C)
        q_dist = q_dist.view(B,N,C)
        return q_dist, sampled_out 

    def generate(self, shape, batch_size=64, canvas=None): #, patch_size=0, canvas_size=0):
        ''' perform categorical sampling 
        Returns: 
            x: B,N,input_dim
        opt: 
            canvas: (2D init canvas, one_hot_prev: selection as last step)
        '''
        visualize_generation = 1
        self.sampler.clean_mem()
        B = batch_size  
        N = shape
        csize = self.canvas_size
        param = next(self.parameters())
        ## xrand = torch.randn((B,self.input_dim)).to(param.device)
        if canvas is None:
            return_canvas = 0
            canvas = torch.zeros((B,self.input_dim,csize,csize)).to(param.device)  # as negative 
            one_hot_prev = torch.zeros((B,self.cat_one_hot_dim)).to(param.device)
        else:
            return_canvas = 1 # return the accumulated canvas 
            canvas, one_hot_prev = canvas 
            B = canvas.shape[0]

        hist = None
        sampled_out_list = []
        visualize_list = []
        canvas_stack = canvas.unsqueeze(1) # B,canvasd,C -> B,1,canvasd,C
        one_hot_prev= one_hot_prev.unsqueeze(1) # B,nD -> B,1,nD
        for n in range(N): # unroll N steps 
            # sampled_out: one-hot encoding, shape: (B,1,ncls_sel+ncls_loc+nstp)
            CHECKSIZE(canvas_stack, (B,n+1,self.input_dim,csize,csize))
            _, sampled_out = self.forward(canvas_stack.view(B*(n+1),self.input_dim,csize,csize), 
                one_hot_prev.view(B*(n+1),self.cat_one_hot_dim), 
                # mask_out_prevloc=self.mask_out_prevloc_samples, 
                is_generate=True, nloc=n+1) 
            CHECKSIZE(sampled_out, (B,n+1,self.cat_one_hot_dim))
            sampled_out = sampled_out[:,-1:] # choose the last one 
            # sampled_out in shape: (B,1,self.cat_one_hot_dim)
            # one_hot_prev: (B,self.cat_one_hot_dim)
            CHECKSIZE(sampled_out, (B,1,self.cat_one_hot_dim))  
            CHECKSIZE(one_hot_prev, (B,n+1,self.cat_one_hot_dim)) 
            one_hot_prev = torch.cat([one_hot_prev, sampled_out], dim=1) # feed as input for next step 
            sampled_out_list.append(sampled_out) # expect shape: B,C 
            CHECKSIZE(sampled_out, (B,1,self.cat_one_hot_dim)) 
            if n == 0: 
                sampled_out = sampled_out ##.unsqueeze(1) # B,1,-1
            else:
                sampled_out = torch.cat(sampled_out_list, dim=1)
            sample_dict = self.sampler.convert_sample_out2dict(sampled_out)
            # here, either choose to accumulate the canvas or accumulate the sample_dict 
            ## sample_dict['init_canvas'] = canvas 

            if self.input_dim > self.imgd:
                canvas_plot, loc_plot = self.canvas_plotter.create_canvas(
                        sample_dict, customize_nloc=n+1, break_in_step=1)
                canvas_plot = canvas_plot.view(B,n+1,self.imgd,csize,csize)
                loc_plot = loc_plot.view(B,n+1,1,csize,csize)
                canvas_plot = torch.cat([canvas_plot, loc_plot], dim=2)[:,-1]
            else:
                canvas_plot = self.canvas_plotter.create_canvas(sample_dict, customize_nloc=n+1, break_in_step=0)
            CHECKSIZE(canvas_plot, canvas)
            
            ### ** used for debugging ** 
            #from torchvision.utils import save_image, make_grid
            #imgd=1 
            #csize=self.canvas_size
            #exp_dir='./exp/cnn_prior/mnist/vis/'
            #save_image(make_grid(canvas.view(B,-1,csize,csize)[0].unsqueeze(1),      nrow=n+1), exp_dir+'/s%02d_input.png'%n)
            #save_image(make_grid(canvas_plot.view(B,-1,csize,csize)[0].unsqueeze(1), nrow=n+1), exp_dir+'/s%02d_add.png'%n)
            #for s, si in sample_dict.items():
            #    logger.info('name: s, {}, value: {} | si: index={}', s, si.shape, si[0,-1].max(0)[1]) ## max(1)[1])

            ## accumulate canvas for cur step 
            canvas = torch.stack([canvas_plot, canvas], dim=0).max(0)[0] # update canvas 
            # canvas_stack: B,n+1,input_dim,csize,csize | canvas: B,input_dim,csize,csize
            canvas_stack = torch.cat([ canvas_stack, canvas.unsqueeze(1) ], dim=1).contiguous()
            #save_image(make_grid(canvas.view(B,2,csize,csize)[:,:1]), exp_dir+'/%02d_acc.png'%n)
            ### x[:,n].copy_(sampled_out[:,n])
        x = torch.cat(sampled_out_list, dim=1) # B,N,D
        if return_canvas:
            return x, canvas
        return x

