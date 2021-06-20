from utils.yacs_config import CfgNode as CN
__C = CN()
cfg = __C
__C.dataset='mnist'

__C.cat_vae=CN()
__C.cat_vae.estimator='gs' 
__C.cat_vae.n_samples=1 
__C.cat_vae.canvas_dim=-1
__C.cat_vae.elbo_weight=1.0

__C.vae = CN() 
__C.vae.layers = 'tanh'
__C.cmt=''
__C.wh=10
__C.ww=10
__C.n_sample=5000
__C.K=5

__C.test_interval=1
__C.use_prior_model=0

__C.model_name='vae'
__C.enc_hid=64
__C.dec_hid=64
__C.latent_d=20 # N, number of latent variable 
__C.stride=4 # down sample twice 

__C.prior_weight='../experiment/patch_prior/models/pc_k50_b32/prior.pt'
__C.use_patch_bank=0
__C.optim = CN() 
__C.optim.diff_lr=0
__C.optim.name='adam'
__C.lr=1e-3
__C.epochs=10
__C.batch_size=128 
__C.test_batch_size=30
__C.test_size=-1
__C.log_interval=500
__C.seed=0 
__C.temp_min=0.5
__C.temp_init=1.0
__C.temp_anneal_rate=3e-5
__C.tag=''
__C.kld_weight=1
__C.bce_weight=1
__C.exp_name='' # set during running
__C.exp_dir=''
__C.exp_key=''
__C.vary_loc_vae=CN() 
__C.vary_loc_vae.enc_hid=0 
__C.vary_loc_vae.canv_d=1 
__C.vary_loc_vae.nloc=5 
__C.vary_loc_vae.pred_locvar=0 
__C.vary_loc_vae.enc_version=-1 
__C.vary_loc_vae.head_version=-1 
__C.vary_loc_vae.latent_loc_weight=1.0
__C.vary_loc_vae.latent_sel_weight=1.0
__C.vary_loc_vae.latent_stop_weight=1.0
__C.vary_loc_vae.te_sel_gt=0
__C.vary_loc_vae.te_sel_gt_temp=0.01
__C.vary_loc_vae.loc_stride=1
__C.vary_loc_vae.stridep=2 # stride = 2^stridep


#__C.prior_model=CN() 
#__C.overwrite=0 # used in create_patches 
#__C.slide_all=0 # 0: sliding all win; 1: slide along edge 
#__C.img_mean=-1
#__C.rank=True
#__C.no_write=0
#__C.cat_vae.hybrid = CN()
#__C.cat_vae.hybrid.latent_dz=20 # N, number of latent variable, for hybrid model
#__C.cat_vae.hybrid.hid_dz=100 # N, number of latent variable, for hybrid model
#__C.cat_vae.hybrid.lw_gau=5.0 # N, number of latent variable, for hybrid model
#__C.cat_vae.hybrid.fusion='early' # [early, late] 
#__C.cat_vae.hybrid.agg='add' # [add, concat] 
#__C.cat_vae.hybrid.use_bn=1 # use bn in encoder or not 
#__C.cat_vae.hybrid.gmm_mix=20 # number of mixture if hybrid with GMM 


for k in ['cat_vae.fixed_loc_dec_version', 'cat_vae.fixed_loc_enc_version', 
    ## 'cat_vae.add_stop', 
    'bn_loss_weight', 'no_cuda', 'input_size', 
    'cat_vae.use_mlp_dec', 'cat_vae.scale_var', 'cat_vae.hybrid', 
    'cat_vae.hybrid.latent_dz', 'cat_vae.hybrid.hid_dz',
    'cat_vae.hybrid.lw_gau',
    'cat_vae.hybrid.fusion',
    'cat_vae.hybrid.agg',
    'cat_vae.hybrid.use_bn',
    'cat_vae.hybrid.gmm_mix',
    'draw',
    'draw.z_size',
    'draw.time_steps', 
    'draw.write_size',
    'draw.read_size',
    'draw.write_atten', 
    'draw.read_atten',
    'draw.lstm_size',
    'require_label2id',
    'prior_hid', 'img_mean', 'train_prior', 'lr_prior', 
    'vary_loc_vae.latent_loc_sample', 'cat_vae.is_pretrain',
    'cat_vae.load_dec', 'cat_vae.log_variance',
    'cat_vae.sample_kl',
    'freeze_encoder',
    'loc_prior',
    'min_loc',
    'minibatches',
    'no_write',
    'nvae_cfg',
    'optim.compute_loss_in_gray_scale',
    'optim.scheduler_name',
    'optim.use_scheduler',
    'optim.sche_steplr',
    'optim.sche_steplr_gamma',
    'overlap',
    'recont_loc',
    'overwrite',
    'rank',
    'slide_all',
    'spe_norm',
    'prior_model',
    'prior_model.eval_with_freq',
    'prior_model.eval_with_freq_loc',
    'prior_model.gt_json',
    'step_size',
    'step_size_recont',
    'valnll_est_nsample', 'vis',
    'supervise_canvas', 'supervise_latent_loc', 
    ]:
    __C.register_deprecated_key(k)
#__C.train_prior=0 
#__C.prior = CN()
#__C.prior.is_picnn_block=0
# patch
#__C.step_size=1
#__C.step_size_recont=1 # for fixed location, step_size_recont is 10.  
__C.register_deprecated_key('hard_patch') 
__C.register_deprecated_key('prior') 
__C.register_deprecated_key('prior.is_picnn_block') 

#__C.spe_norm=0
#__C.loc_prior=0
#__C.prior_model.eval_with_freq=0 # use freq as prior 
#__C.prior_model.eval_with_freq_loc=0 # use freq as prior 
## __C.prior_model.gt_json='' 
__C.register_deprecated_key('center_pat') 

# Patch Prior GT Generation
#__C.min_loc=0 # used for vari-loc prior, require minum patches to be paste 
#__C.overlap='opt11' # selection ['opt11', 'opt31', 'max']
#__C.recont_loc='canny'# selection from {'canny', 'fixed33', 'every_loc'}

# model 
__C.register_deprecated_key('discrete_rv') 
# normalize the alpha value for the dirichlet parameters or not 
#__C.nvae_cfg='./others/NVAE/cifar10_cfg.yml'
__C.register_deprecated_key('NVIL') 
__C.register_deprecated_key('NVIL.id_baseline') 

#__C.NVIL = CN() 
#__C.NVIL.id_baseline=0 

# used in model.pvae 
__C.register_deprecated_key('softmax_activate') 
__C.register_deprecated_key('mlp_pvae')

for k in ['use_patch_bank_prior']: 
    __C.register_deprecated_key(k)
# training 

#__C.optim.scheduler_name='step'
#__C.optim.use_scheduler=0
#__C.optim.sche_steplr=0
#__C.optim.sche_steplr_gamma=0.1
# turn all images into gray scale before computing the recont loss 
# used for model debugging 
#__C.optim.compute_loss_in_gray_scale=0

#__C.lr_prior=3e-4 # used for train prior model 
__C.register_deprecated_key('straight_through')
#__C.minibatches=0 #i.e. number of iteration 
#__C.vis=False 
#__C.valnll_est_nsample=50
#__C.supervise_canvas=0
#__C.supervise_latent_loc=0

__C.register_deprecated_key('hard_sample')
#__C.freeze_encoder=0
#__C.bn_loss_weight=1e-2

__C.register_deprecated_key('select_template_subset') 
__C.register_deprecated_key('norm_recont') 
__C.register_deprecated_key('norm_patch') 
__C.register_deprecated_key('select_all') 
# __C.inverse_train=0 
__C.register_deprecated_key('inverse_train') 

#__C.vary_loc_vae.latent_loc_sample=0
#__C.vary_loc_vae.loc_activate='sigmoid'
#__C.vary_loc_vae.loc_dist='Gaussian'
#__C.vary_loc_vae.post_areg=0 # stride = 2^stridep
#__C.vary_loc_vae.anchor_dependent=0

__C.register_deprecated_key('vary_loc_vae.loc_activate') #'sigmoid'
__C.register_deprecated_key('vary_loc_vae.loc_dist') #'Gaussian'
__C.register_deprecated_key('vary_loc_vae.post_areg') #0 # stride = 2^stridep
__C.register_deprecated_key('vary_loc_vae.anchor_dependent') #0
#__C.eval=CN()
#__C.eval.compute_cluster_acc=1
#__C.eval.plot_tsne=0

__C.register_deprecated_key('eval') #CN()
__C.register_deprecated_key('eval.compute_cluster_acc') #1
__C.register_deprecated_key('eval.plot_tsne') #0
# __C.cat_vae.warm_up=0

__C.register_deprecated_key('cat_vae.warm_up')
#__C.cat_vae.fixed_loc_enc_version=0  
#__C.cat_vae.fixed_loc_dec_version='default'
#__C.cat_vae.load_dec='' 
__C.cat_vae.add_stop=0 # add stop at vary-loc model or not
#__C.cat_vae.log_variance=0 
#__C.cat_vae.sample_kl=0
#__C.cat_vae.use_mlp_dec=0
#__C.cat_vae.is_pretrain=0
#__C.cat_vae.scale_var=1.0
__C.register_deprecated_key('vary_loc_vae.loc_map_in_z')
# __C.cat_vae.eps=1e-6

#__C.cat_vae.hybrid = CN()
#__C.cat_vae.hybrid.latent_dz=20 # N, number of latent variable, for hybrid model
#__C.cat_vae.hybrid.hid_dz=100 # N, number of latent variable, for hybrid model
#__C.cat_vae.hybrid.lw_gau=5.0 # N, number of latent variable, for hybrid model
#__C.cat_vae.hybrid.fusion='early' # [early, late] 
#__C.cat_vae.hybrid.agg='add' # [add, concat] 
#__C.cat_vae.hybrid.use_bn=1 # use bn in encoder or not 
#__C.cat_vae.hybrid.gmm_mix=20 # number of mixture if hybrid with GMM 

#__C.draw = CN() 
#__C.draw.z_size = 100 
#__C.draw.time_steps=64 # follow the setting for MNIST in paper table3
#__C.draw.write_size=5
#__C.draw.read_size=2
#__C.draw.write_atten=1
#__C.draw.read_atten=1
#__C.draw.lstm_size=256
#__C.require_label2id=1 # by default, not given label, but the index
