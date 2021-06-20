from utils.yacs_config import CfgNode as CN
__C = CN()
cfg = __C
# cfg.canvas_init=0
cfg.use_vit=0
cfg.use_fast_vit=0
cfg.img_mean=-1
cfg.vit_mlp_dim=2048
cfg.vit_depth=8
cfg.vit_dropout=1
cfg.concat_one_hot=0
cfg.mask_out_prevloc_samples=0
#cfg.input_id_canvas=0
cfg.register_deprecated_key('input_id_canvas') 
cfg.use_cnn_process=0
cfg.input_id_only=0
cfg.cond_on_loc=0
cfg.gt_file=''
cfg.img_size=28 
cfg.pw=10 
cfg.register_renamed_key('ps', 'pw')
cfg.register_deprecated_key('steps') 
cfg.register_deprecated_key('canvas_init') 
cfg.register_deprecated_key('lw') 
cfg.register_deprecated_key('anchor_dependent') 
cfg.hid=256 
cfg.batch_size=128 
cfg.num_epochs=50 
cfg.lr=3e-4 
## cfg.lw=1.0
cfg.k=50 
cfg.loc_loss_weight=1.0
cfg.cls_loss_weight=1.0 
cfg.stp_loss_weight=1.0 
cfg.output_folder='./exp/prior'
cfg.single_sample=0
cfg.dataset='mnist' 
cfg.add_empty=0 
cfg.add_stop=0
cfg.inputd=2
cfg.model_name='cnn_prior'
cfg.hidden_size_prior=64 
cfg.hidden_size_vae=256 
cfg.use_scheduler=0
cfg.early_stopping=0
cfg.loc_map=1
cfg.nloc=-1
cfg.num_layers=8 #15
cfg.loc_dist='Gaussian' 
cfg.loc_stride=1
cfg.exp_key=''
cfg.device='cuda'
cfg.exp_dir='./exp/' # root of all experiments 
cfg.mhead=0
cfg.kernel_size=7 # for picnn's kernel 
cfg.permute_order=0 # for picnn's kernel 
cfg.geometric=0
#cfg.anchor_dependent=0
cfg.start_time=''
cfg.pos_encode=0
cfg.use_emb_enc=0
