"""
optimizer, Scheduler related 
"""
from loguru import logger 
import torch 
from torch.autograd import grad
MODEL_LIST_NO_CANV = ['vae', 'cvae', 'cvae2', 'cvaer', 'srvae', 'vaedense', 'cvaedc']
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist 
class obj(object):
    def __init__(self, d): 
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)

#def get_ddpm_config(cfg, image_size, in_channels=3):
#    type = "simple"
#    # in_channels = 3
#    out_ch = 3
#    ch = 128
#    num_res_blocks = 2
#    attn_resolutions = [16, ]
#    resamp_with_conv = True
#    if 'cifar' in cfg.dataset:
#        ch_mult = [1, 2, 2, 2]
#        dropout = 0.1
#        var_type = 'fixedlarge'
#        ema_rate = 0.9999
#    elif 'celeba' in cfg.dataset:
#        ch_mult = [1, 2, 2, 2, 4]
#        dropout = 0.1
#        var_type = 'fixedlarge'
#        ema_rate = 0.9999
#    else:
#        raise NotImplementedError
#    #elif 'bedroom' in cfg.dataset:
#    #    ch_mult = [1, 1, 2, 2, 4, 4]
#    #    dropout = 0.0
#    #    var_type = 'fixedsmall'
#    #    ema_rate = 0.999
#    #elif 'church' in cfg.dataset:
#    #    ch_mult = [1, 1, 2, 2, 4, 4]
#    #    dropout = 0.0
#    #    var_type = 'fixedsmall'
#    #    ema_rate = 0.999
#    config = {'type': type, 'in_channels': in_channels,
#        'out_ch': out_ch, 'ch': ch, 'num_res_blocks': num_res_blocks,
#        'attn_resolutions': attn_resolutions, 'ch_mult': ch_mult,
#        'dropout': dropout, 'resamp_with_conv':resamp_with_conv}
#    return obj({
#        'model': config, 
#        'diffusion':{
#            'num_diffusion_timesteps': cfg.diffu.max_step 
#            }, 
#        'data': {'image_size': image_size} 
#        })
#
#
#def aggregate_loss(cfg, output, metric, names=None): 
#    loss = 0 
#    if names is not None: 
#        assert(names in output), f'{names} not in {list(output.keys())}'
#        metric.update(BCE=output[names].mean(0)) 
#        loss += output[names].mean() # shape: B,1 
#    if 'bp_loss' in output:
#        # -- use bp_loss only -- #
#        metric.update(canvas_loss=output['bp_loss'].mean(0)) 
#        loss += output['bp_loss'].mean() # shape: B,1 
#        return loss 
#    if 'ELBO' in output:
#        loss += output['ELBO'].mean(0) * cfg.cat_vae.elbo_weight
#        self.metric.update(BCE=output_dict['BCE'].mean(0), KL=output_dict['KLD'].mean(0), ELBO=output['ELBO'].mean(0))
#
#    if 'canvas_loss' in output:
#        metric.update(canvas_loss=output['canvas_loss'].mean(0)) 
#        loss += output['canvas_loss'].mean() # shape: B,1 
#    if 'loc_loss' in output: 
#        metric.update(loc_loss=output['loc_loss'].mean(0)) 
#        loss += output['loc_loss'].mean() * cfg.vary_loc_vae.latent_loc_weight 
#    if 'sel_loss' in output: 
#        metric.update(sel_loss=output['sel_loss'].mean(0)) 
#        loss += output['sel_loss'].mean() * cfg.vary_loc_vae.latent_sel_weight 
#    if 'stp_loss' in output: 
#        metric.update(stp_loss=output['stp_loss'].mean(0)) 
#        loss += output['stp_loss'].mean() * cfg.vary_loc_vae.latent_sel_weight 
#    if 'bn_loss' in output: 
#        metric.update(bn_loss=output['bn_loss'].mean(0)) 
#        loss += output['bn_loss'].mean() * cfg.bn_loss_weight 
#    if 'sn_loss' in output: 
#        metric.update(sn_loss=output['sn_loss'].mean(0)) 
#        loss += output['sn_loss'].mean() * cfg.bn_loss_weight 
#    if 'diffu_loss' in output: 
#        metric.update(diffu_loss=output['diffu_loss'].mean(0)) 
#        loss += output['diffu_loss'].mean() #* cfg.bn_loss_weight 
#    return loss
#def aggregate_loss_rl(cfg, output, metric): # for reinforce_bl estimator 
#    loss = 0
#    if 'loc_loss' in output: 
#        metric.update(loc_loss=output['loc_loss'].mean(0)) 
#        loss += output['loc_loss'].mean() * cfg.vary_loc_vae.latent_loc_weight
#    if 'sel_loss' in output: 
#        metric.update(sel_loss=output['sel_loss'].mean(0)) 
#        loss += output['sel_loss'].mean() * cfg.vary_loc_vae.latent_sel_weight
#    if 'stp_loss' in output: 
#        metric.update(stp_loss=output['stp_loss'].mean(0)) 
#        loss += output['stp_loss'].mean() * cfg.vary_loc_vae.latent_sel_weight 
#    return loss


class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    # DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


#def clip_grad(optimizer):
#    with torch.no_grad():
#        for group in optimizer.param_groups:
#            for p in group['params']:
#                state = optimizer.state[p]
#
#                if 'step' not in state or state['step'] < 1:
#                    continue
#
#                step = state['step']
#                exp_avg_sq = state['exp_avg_sq']
#                _, beta2 = group['betas']
#
#                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
#                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))
#

#
def average_gradients(model, rank=-1):
    size = float(dist.get_world_size())
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        param.grad.data /= size
    torch.cuda.synchronize()
#
## ----- Scheduler -----
#class LowerBoundedExponentialLR(_LRScheduler):
#    def __init__(self, optimizer, gamma, lower_bound, last_epoch=-1):
#        self.gamma = gamma
#        self.lower_bound = lower_bound
#        super(LowerBoundedExponentialLR, self).__init__(optimizer, last_epoch)
#
#    def _get_lr(self, base_lr):
#        lr = base_lr * self.gamma ** self.last_epoch
#        if lr < self.lower_bound:
#            lr = self.lower_bound
#        return lr
#
#    def get_lr(self):
#        return [self._get_lr(base_lr)
#                for base_lr in self.base_lrs]
#
#
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) ## if p.requires_grad)

def freeze_parameter(model):
    logger.info('-- [freeze_parameter] --')
    for name, param in model.named_parameters():
        param.requires_grad = False 
#
#
#def check_grad(loss, x):
#    loss = loss.sum() 
#    d_loss_dx = grad(outputs=loss, inputs=x, retain_graph=True)
#    logger.info(f'dloss/dx:\n {d_loss_dx}')
#
#def is_bad_grad(grad_output):
#    if grad_output.requires_grad == False:
#        logger.info('grad_ouput doesnt have grad')
#    grad_output = grad_output.data 
#    return grad_output
#
#def get_optim(cfg, parameters, named_parameters, lr): 
#    if not cfg.optim.diff_lr and not cfg.freeze_encoder:
#        return torch.optim.Adam(parameters, lr)
#    else: 
#        group_normal, group_small = [], []
#        if cfg.optim.diff_lr:
#            key_small = ['decoder']
#            lr_small, lr_norm = 0.1*lr, lr 
#        if cfg.freeze_encoder:
#            lr_norm = 0.0
#        for name, param in named_parameters: 
#            if not param.requires_grad: continue 
#            assert('prior' not in name), f'found {name} require grad '
#            need_reduce = sum([int(k in name) for k in key_small])
#            if need_reduce: 
#                group_small.append(param) 
#                logger.debug('LR for {}: {:.5f}', name, lr_small)
#            else:
#                group_normal.append(param)
#                logger.debug('LR for {}: {:.5f}', name, lr_norm)
#        opt_list = []
#        if lr_norm > 0:
#            opt_list.append( {'params': group_normal, 'lr': lr_norm} )
#        opt_list.append(
#           {'params': group_small, 'lr': lr_small} 
#           )
#        logger.info('[Build Optimizer] #group={}', len(opt_list)) 
#        if cfg.optim.name == 'adam':
#            opt = torch.optim.Adam( opt_list )
#        elif cfg.optim.name == 'adamax':
#            opt = torch.optim.Adamax( opt_list, betas=(0.9, 0.999), eps=1e-7)
#        else:
#            raise ValueError(cfg.optim.name)
#        return opt 
#
