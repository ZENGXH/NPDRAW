python train_vae.py --comet 1 \
    vary_loc_vae.te_sel_gt 4 vary_loc_vae.te_sel_gt_temp 0.005 \
    cmt "mnist" \
    vary_loc_vae.post_areg 0 vary_loc_vae.loc_dist cat cat_vae.add_stop 1 \
    optim.diff_lr 1 bce_weight 1 kld_weight 1 test_batch_size 20 \
    model_name cat_vloc_at cat_vae.hybrid.latent_dz 0 seed 0 epochs 661 \
    tag p5s5n36vitBinkl1r lr 1e-3 batch_size 500 dataset stoch_mnist \
    K 50 dec_hid 128 enc_hid 128 wh 5 ww 5 supervise_latent_loc 1 \
    cat_vae.canvas_dim 1 \
    supervise_canvas 0 temp_init 1.0 overlap max temp_min 0.5 slide_all 0 \
    use_patch_bank 1 use_patch_bank_prior 1 use_prior_model 1 temp_anneal_rate 0.013862944 \
    prior_weight exp/cnn_prior/mnist/0208/sd5_vitcnnS2c_binm1E4b64h64n36p5_reg_aspsc/prior.pt \
    cat_vae.estimator gs vary_loc_vae.stridep 2 vary_loc_vae.nloc 36 \
    vary_loc_vae.latent_loc_weight 50.0 vary_loc_vae.latent_sel_weight 50.0 \
    vary_loc_vae.latent_stop_weight 1.0 vary_loc_vae.loc_stride 5
