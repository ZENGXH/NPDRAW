#!/usr/bin/env python
# coding: utf-8
import os 
import torch 
from torch.nn import functional as F
from torchvision import datasets 
import torch.utils.data as data
import numpy    as np
from PIL import Image 
from loguru import logger
from utils.checker import *
from tqdm import tqdm 

# ------ copy from residual flow /lib/datasets.py
DATAROOT='./datasets/'
class CelebA5bit(object):
    ''' celebahq64_5bit 
    from celeba hq, comparessed into 5 bit data 
    torch.Size([26838, 3, 64, 64]), value range: [0,1,..,31], dtype=torch.uint8
    '''
    def __init__(self, DATAROOT, train=True, transform=None, download=None):
        self.LOC = '%s/celebahq64_5bit/celeba_full_64x64_5bit.pth'%DATAROOT
        self.dataset = torch.load(self.LOC)
        self.dataset = torch.cat([self.dataset[:,2:3], self.dataset[:,1:2], self.dataset[:,0:1]], dim=1)
        logger.info('data set loaded: {}, min={}, max={}', self.dataset.shape, self.dataset.min(), self.dataset.max()) 
        
        self.dataset = self.dataset.float() / 31.0 # range [0,1] 
        if not train:
            self.dataset = self.dataset[:5000] # eval, use the first 5k images 
        else:
            self.dataset = self.dataset[5000:] # train, use images starting from 5k 
        self.transform = transform
        logger.info('data trainsform: {}', transform) 
    def __len__(self):
        return self.dataset.size(0)
    @property
    def ndim(self):
        return self.dataset.size(1)
    def __getitem__(self, index): 
        x = self.dataset[index] # 3,H,W
        x = self.transform(x) if self.transform is not None else x
        return x, index 

    @property
    def images(self):
        # the original dataset is compressed into 5 bit, which used to be 8 bits. 
        return (self.dataset*255.0).to(torch.uint8).float() # 0-31 -> 0-1 -> 0-255
    
    def label2imgid(self):
        return True 
class RFDataset(object):
    def __init__(self, loc, transform=None, in_mem=True, download=None):
        self.in_mem = in_mem
        logger.info('[Load] {} | {}', loc, in_mem)
        self.dataset = torch.load(loc)
        if in_mem: self.dataset = self.dataset.float().div(255)
        self.images = self.dataset  
        self.transform = transform
        logger.info('[Build RFDataset] shape={}, transform={} | max={:.1f} | type={}', 
            self.dataset.shape, self.transform, self.dataset.max(), self.dataset.dtype)
    def __len__(self):
        return self.dataset.size(0)
    @property
    def ndim(self):
        return self.dataset.size(1)
    def __getitem__(self, index):
        x = self.dataset[index]
        if not self.in_mem: x = x.float().div(255)
        x = self.transform(x) if self.transform is not None else x
        return x, index 

    def label2imgid(self):
        return True

class CelebAHQ(data.Dataset):
    TRAIN_LOC = f'{DATAROOT}/celebahq_img/train/%d.png'
    TEST_LOC = f'{DATAROOT}/celebahq_img/valid/%d.png'
    def __init__(self, DATAROOT, split='train', transform=None, download=None):
        ''' dataset, images: N,D,H,W (train:27000,3,256,256) '''
        super().__init__()
        train = split == 'train'
        self.train = train 
        # super(CelebAHQ, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform, in_mem=in_mem)
        # self.images = self.images.permute(0,2,3,1) # NHWD 
        if self.train: 
            self.root = self.TRAIN_LOC 
        else:
            self.root = self.TEST_LOC 
        self.transform = transform

    def label2imgid(self):
        return True
    def __len__(self):
        return 27000 if self.train else 3000  

    def __getitem__(self, index):
        # DHW
        img = Image.open(self.root%index)
        # x = np.array(img)
        x = img 
        ## x = self.dataset[index] ## .permute(2,0,1) # HWD -> DHW
        x = self.transform(x) if self.transform is not None else x
        return x, index  

class CelebAHQ_PTH(RFDataset):
    TRAIN_LOC = f'{DATAROOT}/celebahq/celeba256_train.pth'
    TEST_LOC = f'{DATAROOT}/celebahq/celeba256_validation.pth'
    def __init__(self, DATAROOT, split='train', transform=None, download=None, in_mem=False):
        ''' dataset, images: N,D,H,W (train:27000,3,256,256) 
        '''
        train = split == 'train'
        super().__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform, in_mem=in_mem)
        self.images = self.images.permute(0,2,3,1) # NHWD 
    def __getitem__(self, index):
        # DHW
        x = self.dataset[index] ## .permute(2,0,1) # HWD -> DHW
        x = self.transform(x) if self.transform is not None else x
        return x, index  

    #def save_into_split(self):
    #    ns = int ( len(self.dataset) // 3 )
    #    for i in range(3):
    #        TRAIN_LOC_SUB = f'{DATAROOT}/celebahq/celeba256_train-{i}.pth'
    #        dataset_sub = self.dataset[i*ns:i*ns+ns] 
    #        dataset_sub_sto = dataset_sub.size * dataset_sub.itemsize * 1e-9 
    #        logger.info('dataset_sub: {} | {}; size={:.1f}G | {}', 
    #            dataset_sub.shape, TRAIN_LOC_SUB, dataset_sub_sto, type(dataset_sub))
    #        torch.save(dataset_sub, TRAIN_LOC_SUB)

class Imagenet32(RFDataset):
    TRAIN_LOC = f'{DATAROOT}/imagenet32/train_32x32.pth'
    TEST_LOC = f'{DATAROOT}/imagenet32/valid_32x32.pth'
    def __init__(self, train=True, transform=None, download=None):
        return super(Imagenet32, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform)

class Imagenet64(RFDataset):
    TRAIN_LOC = f'{DATAROOT}/imagenet64/train_64x64.pth'
    TEST_LOC = f'{DATAROOT}/imagenet64/valid_64x64.pth'
    def __init__(self, train=True, transform=None):
        return super(Imagenet64, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform, in_mem=False)
# ------ end of dataset 

class CifarFews(datasets.CIFAR10):
    # will load n% of the train set as training data, 1-n% of the train set as validation data 
    # will not used the val set of the original cifar10 dataset 
    def __init__(self, percent, dataroot, **kwargs):
        train_train = kwargs['train'] 
        kwargs['train'] = True # force to be true 
        super().__init__(dataroot, **kwargs)
        n_train_ori = len(self.data) 
        n_train_new = int(n_train_ori * percent)
        logger.info('[Build CIFAR Low Data] use {} for train. {}/{}', percent,
            n_train_new, n_train_ori)
        assert(percent < 1.0), 'only support float'
        if train_train:
            self.data = self.data[:n_train_new]
            self.targets = self.targets[:n_train_new]
        else:
            nval = min(1000, len(self.data)-n_train_new)
            self.data = self.data[n_train_new:n_train_new+nval]
            self.targets = self.targets[n_train_new:n_train_new+nval]
if __name__ == '__main__':
    for split in ['valid', 'train']:
        data = CelebAHQ_PTH('datasets', split=split, in_mem=False)
        root = f"./datasets/celebahq_img/{split}/%d.png"
        if not os.path.exists(os.path.dirname(root)):
            os.makedirs(os.path.dirname(root))
            logger.info('create dir: {}', os.path.dirname(root))
        for img, index in tqdm(data): 
            img = Image.fromarray( img.permute(1,2,0).numpy() )  # 3HW -> HWD 
            img.save(root%index) 

    ## data.save_into_split()
