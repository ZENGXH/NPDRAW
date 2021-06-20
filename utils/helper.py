#!/usr/bin/env python
# coding: utf-8
import cv2
from copy import deepcopy
import os
import sys
import yaml 
import time
import torch 
from torch.nn import functional as F
import pickle 
from tqdm import tqdm
from scipy import ndimage 
import matplotlib.pyplot as plt
# import torchvision.datasets 
from torchvision import datasets, transforms
import numpy    as np
from PIL import Image, ImageDraw
from torchvision.utils import save_image, make_grid
import matplotlib.patches as patches
from loguru import logger
from utils import data_helper
from utils.checker import *
from PIL import Image
from matplotlib import cm
from functools import partial
import re 

sliding_method = ['sliding_along_edge', 'sliding_window', 'uniform_window', 
    'sliding_at_kept', 'sliding_at_nonzero'] 

def parse_subset_size_0to1(dataset_name):
    '''
    parse the percent of images, 
    before: 
    01 -> 1% of full train data, return 0.01 
    10 -> 10% of full train data, return 0.1
    now add support: 
    001 -> 0.1% of full train data, return 0.001
    not support: 1: (wgan) which is ambiguous 
    '''
    if re.search('cifar([\d]+)', dataset_name): 
        percent_str = re.search('cifar([\d]+)', dataset_name).group(0).split('cifar')[-1] 
        assert( len(percent_str) >= 2), 'require to has length at least 2'
        percent_float = int(percent_str) / (10**len(percent_str))
    elif re.search('celebaf([\d]+)', dataset_name): 
        percent_str = re.search('celebaf([\d]+)', dataset_name).group(0).split('celebaf')[-1] 
        assert( len(percent_str) >= 2), 'require to has length at least 2'
        percent_float = int(percent_str) / (10**len(percent_str))
    elif re.search('celebaCr148f([\d]+)', dataset_name): 
        percent_str = re.search('celebaCr148f([\d]+)', dataset_name).group(0).split('celebaCr148f')[-1] 
        assert( len(percent_str) >= 2), 'require to has length at least 2'
        percent_float = int(percent_str) / (10**len(percent_str))
    elif re.search('mnistf([\d]+)', dataset_name):
        percent_str = re.search('mnistf([\d]+)', dataset_name).group(0).split('mnistf')[-1] 
        assert( len(percent_str) >= 2), 'require to has length at least 2'
        percent_float = int(percent_str) / (10**len(percent_str))
    elif re.search('omnif([\d]+)', dataset_name):
        percent_str = re.search('omnif([\d]+)', dataset_name).group(0).split('omnif')[-1] 
        assert( len(percent_str) >= 2), 'require to has length at least 2'
        percent_float = int(percent_str) / (10**len(percent_str))
    else:
        raise ValueError(dataset_name)
    return percent_float
def build_data_set(dataset_name, istrain, cust_trans=None): 
    '''
    omni_one_shot VS omni: 
        the former one has 30 alphabets in background(train) set and 20 alphabets in eval set; 
        the number of samples in train and eval is about 10k VS 10k
        while the later one has 8k in eval set, and about 30k in train set, the split follows the IWAE paper 
    '''
    eargs_te, eargs_tr = {}, {}
    if 'celeba' in dataset_name: # crop at 148x148
        eargs_te['split'] = 'valid' #test'
        eargs_tr['split'] = 'train'
    else: # 'mnist' in dataset_name:
        eargs_te['train'] = False
        eargs_tr['train'] = True
    T = transforms.ToTensor()
    if cust_trans and not dataset_name == 'omni32':
        assert ValueError('only omni32 support cust_trans')

    if dataset_name == 'mnist':
        logger.info('use datasets.MNIST obj')
        data_obj = datasets.MNIST 
    elif dataset_name == 'stoch_mnist':
        from utils.stoch_mnist import stochMNIST 
        data_obj = stochMNIST 
    elif 'mnistf' in dataset_name:
        if 'v' in dataset_name:
            split_index = int(re.findall('mnistf([\d]+)v([\d]+)', dataset_name)[0][1])
        else:
            split_index = 0
        # percent = int(percent) / 100.0
        percent = parse_subset_size_0to1(dataset_name)
        logger.debug('build mnist few shot with name: {} | create partial obj, per={},splitID={}', 
            dataset_name, percent, split_index)
        if 'dmnist' in dataset_name:
            from utils.stoch_mnist import MNISTfew 
            assert(percent >= 0.1), 'accept percent in 0.1,0.2,0.3,...0.9,1 only, get %f'%percent
        else: ## if 'smnist' in dataset_name:
            from utils.stoch_mnist import MNISTfewBySample as MNISTfew
            if split_index > 0: raise NotImplementedError('not support index > 0')
        data_obj = partial(MNISTfew, percent, split_index)
    elif dataset_name == 'fixed_mnist':
        from utils.stoch_mnist import fixedMNIST 
        data_obj = fixedMNIST 
    # -------------------------------
    # low_data regime, for omniglot  
    elif 'omnif' in dataset_name:
        percent = parse_subset_size_0to1(dataset_name) 
        # int(dataset_name.split('omnif')[-1]) / 100.0 
        logger.debug('build omni few shot with name: {} | create partial obj', dataset_name)
        if 'aomnif' in dataset_name:
            # split by alphabet 
            from utils.omniglot import omniglot_fews_alphabet as omniglot_fews 
            if re.search('aomnif([\d]+)v([\d])',dataset_name):
                random_split_index = int(re.findall('aomnif([\d]+)v([\d])', dataset_name)[0][1])
                logger.info('[build_data_set] get random split index: {}', random_split_index)
                if random_split_index != 0: raise NotImplementedError('not support index > 0 now')

            elif re.search('aomnif([\d]+)',dataset_name):
                random_split_index = 0 ## int(re.findall('aomnif([\d]+)', dataset_name)[0])
                logger.info('[build_data_set] get random split index: {}', random_split_index)
                if random_split_index != 0: raise NotImplementedError('not support index > 0 now')
        else:
            from utils.omniglot import omniglot_fews 
        data_obj = partial(omniglot_fews, percent)
    # ------------------------------
    elif dataset_name == 'omni': 
        from utils.omniglot import omniglot 
        data_obj = omniglot 
    elif dataset_name in ['cifar', 'cifarg', 'cifargs', 'cifarc', 'cifarcm', 'cifarcs',
            'cifarc2s']: 
        data_obj = datasets.CIFAR10 
    elif re.search('cifar([\d]+)', dataset_name): 
        from utils.datasets import CifarFews 
        percent = parse_subset_size_0to1(dataset_name)
        # e.g., cifar90, cifar90c, cifar90g, cifar90gs, 
        data_obj = partial(CifarFews, percent) 
    elif re.search('celebaf([\d]+)', dataset_name): 
        logger.info('BUILD data: tag=celeba few ')
        from utils.celeba import CelebAFews 
        percent_float = parse_subset_size_0to1(dataset_name) 
        # e.g., cifar90, cifar90c, cifar90g, cifar90gs, 
        data_obj = partial(CelebAFews, percent_float)
        T = get_data_transforms(dataset_name, istrain)
        logger.debug('data: {}, transform: {}', dataset_name, T)
    elif 'celeba' in dataset_name:
        logger.info('BUILD data: tag=celeba')
        from utils.celeba import CelebA 
        data_obj = CelebA 
        T = get_data_transforms(dataset_name, istrain)
        logger.debug('data: {}, transform: {}', dataset_name, T)
    else:
        raise ValueError('NOT support %s'%dataset_name)
    logger.debug('data_obj: {} | tr: {}, te: {}', data_obj, eargs_tr, eargs_te)
    if istrain: 
        loaded_set = data_obj('datasets', download=True,
                       transform=T, **eargs_tr)
    else:
        loaded_set = data_obj('datasets', download=True,
                transform=T, **eargs_te)
    logger.info('<dataset> {} (n={}) is built', dataset_name, len(loaded_set))
    return loaded_set 

def get_data_transforms(dataset_name, istrain):
    # SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    # SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
    tlist = []
    if 'celeba' in dataset_name: # follow realNVP, use 148 cropping  
        img_size = data_helper.get_imgsize(dataset_name) 
        ## crop_size = 140 if '40' in dataset_name else 148
        crop_size = data_helper.get_cropsize(dataset_name) ## 140 if '40' in dataset else 148 
        tlist.extend([
            transforms.CenterCrop(crop_size),
            transforms.Resize(img_size),
            transforms.ToTensor()])
        transform = transforms.Compose(tlist)
    else:
        raise NotImplementedError 
    return transform 

