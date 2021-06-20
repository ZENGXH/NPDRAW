# -*- coding: utf-8 -*-
import sys 
import os
import yaml
import numpy as np
from loguru import logger 
from datetime import datetime
from comet_ml import Experiment
from runner.classifier import classifier_tsne as TestRunner 
from runner.category_vae_runner import DiscreteRunner as DisRunner 
from utils.metric_logger import MetricLogger 
from config import cfg
import torch 
import argparse 
from utils import io_helper

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

parser = argparse.ArgumentParser(description="args of train_vae.py")
parser.add_argument("--comet", help="disable comet or not: 0 - disable, 1 - enable, 2 - offline", default=1, type=int)  
parser.add_argument("--resume", help="path of ckp to load, resume from the exp in the folder, use the # cfg's exp key and load the last checkpoint model", default=None, type=str)
parser.add_argument("--eval_only", help="no training", default=0, type=int) 
parser.add_argument("--eval_fid_only", help="generate 10k sample for fid ", default=0, type=int) 
parser.add_argument('--local_rank', dest='local_rank',default = 0, type=int)
parser.add_argument('--distributed', dest='distributed',default=False, type=bool)
parser.add_argument('--ngpu', dest='ngpu',default=1, type=int)
parser.add_argument( "opts", help="Modify config options using the command-line", 
    default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

logger.info('get args: distributed=%d '%(args.distributed)) 
logger.info('Train VAE, rank=%d'%args.local_rank) 

comet_disable=args.comet == 0 
if args.local_rank > 0:
    comet_disable = True 

if args.comet:
    exp = Experiment(project_name="patch2img", auto_param_logging=False, 
        auto_metric_logging=False, parse_args=False,
        display_summary_level=0, disabled=comet_disable)
else:
    exp = None

# @logger.catch(reraise=True) 
def main():
    # merge args to config
    if args.resume is not None: # resume from  
        logger.info('resuming from {}, dir name {}', args.resume, 
            os.path.dirname(args.resume))
        resume_cfg_p = os.path.join(os.path.dirname(args.resume), 'cfg.yml') 
        cfg.merge_from_file(resume_cfg_p)
    cfg.merge_from_list(args.opts) 
    # create exp_name 
    exp_name = io_helper.create_exp_name(cfg)
    # ------------------------------------------------------ 
    SEED = cfg.seed 
    np.random.seed(SEED)
    torch.manual_seed(SEED) 
    EXP_ROOT = './exp'
    exp_dir = os.path.join(EXP_ROOT, cfg.dataset, cfg.model_name)

    if args.resume is None and not args.eval_only: 
        cfg.exp_name = exp_name 
    if exp is not None:
        exp.set_name(cfg.exp_name) 

    cfg.exp_dir  = exp_dir
    cfg.exp_key  = exp.get_key() if exp is not None else ''
    # some assertion for the config: 
    full_exp_dir = os.path.join(exp_dir, cfg.exp_name)
    if not os.path.isdir(full_exp_dir) and args.local_rank == 0: 
        os.makedirs(full_exp_dir) 
    if args.local_rank == 0:
        log_file = 'eval.log' if args.eval_only else 'train.log'
        log_file = os.path.join(exp_dir, cfg.exp_name, log_file)
        logger.add(log_file, colorize=False, level='TRACE') 
        logger.info('log output: {} | exp name: {}; exp key: {}', log_file, 
            cfg.exp_name, cfg.exp_key)
              
    # ---------- write to the slurm dir with resume.sh ----------- #
    metric = MetricLogger() 
    if exp is not None: 
        metric.reg_exp(exp)
        metric.log_hp(cfg)
    metric.set_exp_dir(os.path.join(exp_dir, cfg.exp_name))
    
    if args.eval_only:  
        Runner = TestRunner 
        logger.debug('Build Runner as Clsssifier')
    else: 
        Runner = DisRunner 
    cfg.distributed = args.distributed 
    runner = Runner(cfg, metric, args.local_rank, sample_only=args.eval_fid_only) 
    logger.info('BUILD Model: {}', runner.model)
    # ------ Load Models Weight if needed ------ 
    if args.resume is not None: # Resume training  
        ckpt = torch.load(args.resume, map_location=runner.device)
        runner.load_model(ckpt) # also load epoch, test_loss_best 
        logger.info('-'*5 + 'load ckpt from %s; epoch=%d'%(args.resume, ckpt['epo']) + '-'*5)
        if not args.eval_only: # load other training component : optim, scheduler
            logger.info('='*5 + 'load optim from %s'%args.resume + '='*5)
            logger.info('optim groups {}', len(ckpt['optim']))
            ratio = cfg.lr / ckpt['optim']['param_groups'][0]['lr'] 
            for p in ckpt['optim']['param_groups']:
                p['lr'] = p['lr'] * ratio
                logger.info('optim LR changed*{} {} ', ratio, p['lr'])
            runner.optimizer.load_state_dict(ckpt['optim']) 

    if not args.local_rank: 
        logger.info('\n'+metric.comet_url)
        logger.info('\n' + '*'*10 + f'\n{runner.cfg.exp_key}') 
    
    if args.eval_fid_only: 
        runner.sample_10k(ckpt['epo']-1)
    elif args.eval_only: # compute ELBO  
        logger.info('test batch_size: {}', runner.full_test_loader.batch_size)
        runner.test(ckpt['epo'], 50) 
    else: # train 
        runner.train_epochs()
        logger.info('compute elbo')
        runner.compute_elbo_nsample(cfg.epochs)

if __name__ == '__main__':
    main()
