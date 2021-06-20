'''
source: 
    https://github.com/yoonholee/pytorch-vae/blob/master/data_loader/stoch_mnist.py 

Quantitative results on dynamically binarized MNIST

| Method  | NLL (this repo) | NLL ([IWAE paper](https://arxiv.org/abs/1509.00519)) | NLL ([MIWAE paper](https://arxiv.org/abs/1802.04537)) | comments |
| ------------- | ------------- | ------------- | ------------- | ---- |
| VAE | 87.01 | 86.76 | - |
| MIWAE(5, 1) | 86.45 | 86.47 | - | listed as VAE with k=5
| MIWAE(1, 5) | 85.18 | 85.54 | - | listed as IWAE with k=5
| MIWAE(64, 1) | 86.07 | - | 86.21 | listed as VAE 
'''
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from loguru import logger 
import os
  
class stochMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.label2imgid = False 

    def set_label2imgid(self, flag=True):
        self.label2imgid = flag 
 
    """ Gets a new stochastic binarization of MNIST at each call. """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img)
        img = torch.bernoulli(img)  # stochastically binarize
        if self.label2imgid:
            return img, index 
        else:
            return img, target
    
    def get_mean_img(self):
        imgs = self.train_data.type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img


