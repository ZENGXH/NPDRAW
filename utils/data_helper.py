import re 
from loguru import logger
import torch 
import torch.functional as F 
from scipy import ndimage 
from copy import deepcopy
import re 
def get_cover(patch_size, img_size): 
    # by default: it should return the patch_size
    # get the cover size (used to compute the loc gaussian_map 
    # return patch_size if patch_size is 5 or 10 
    # other wise return 10 
    if patch_size < img_size: 
        return patch_size 
    else:
        return 10 

def get_img_mean(dataset):
    if 'celeba' in dataset and 'g' not in dataset:
        return 111 
    elif 'cifar' in dataset: # and 'g' not in dataset:
        return 120
    else:
        return 0
def get_test_data(dataset):
    if 'omnif' in dataset: 
        td = 'omni'
    elif re.search('cifar[0-9][0-9]cm', dataset): 
        td = 'cifarcm'
    elif re.search('cifar[0-9][0-9]c', dataset): 
        td = 'cifarc'
    elif re.search('cifar[0-9][0-9]', dataset): 
        td = 'cifar'
    elif re.search('celebaf([\d]+)', dataset): 
        td = 'celeba'
    elif re.search('mnistf([\d]+)', dataset): 
        td = 'stoch_mnist'
    #elif re.search('celeba[0-9][0-9]c', dataset): # can't do this, since we have celeba40 and celeba14 before 
    #    td = 'celebac'
    #elif re.search('celeba[0-9][0-9]', dataset): 
    #    td = 'celeba'
    else:
        td = dataset 
    return td 

# convert label to imgid, used to index the gt 
def label2imgid(mnist_set): 
    for k in range(len(mnist_set)): 
        invert_op = getattr(mnist_set, "label2imgid", None)

        if callable(invert_op):
            mnist_set.label2imgid() # set imgid as label 
        elif 'targets' in mnist_set.__dict__:
            mnist_set.targets[k] = k
        elif 'labels' in mnist_set.__dict__:
            mnist_set.labels[k] = k
        else:
            logger.info('get data_set entry: {}, {}', mnist_set.__dict__.keys(), mnist_set)
            raise NotImplementedError

    return mnist_set 

# input = rgb2gray(input) 
def rgb2gray(rgb):
    assert(rgb.shape[-1] == 3), rgb.shape 
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray 

class Max2DAddaptive(object):
    def __init__(self, output_size):
        self.pool = torch.nn.AdaptiveMaxPool2d(output_size)
        self.output_size = output_size
    def __call__(self, input):
        # valid input 
        out = self.pool(input) #F.max_pool2d(input, 2, stride=2,padding=0)
        return out
    def __repr__(self):
        return self.__class__.__name__ + f'into {self.output_size}' 


class Max2D(object):
    def __init__(self):
        self.pool = torch.nn.MaxPool2d(2,stride=2, padding=0) 

    def __call__(self, input):
        # valid input 
        out = self.pool(input) #F.max_pool2d(input, 2, stride=2,padding=0)
        return out
    
    def __repr__(self):
        return self.__class__.__name__ + '(2x2,s=2,p=0),out=floor(0.5*in)'


class StochBin(object):
    def __call__(self, img):
        # valid input 
        assert(not img.max() > 1 and not img.min() < 0), \
                'image range: {} TO {}'.format(img.min(), img.max()) 
        img = torch.bernoulli(img)  # stochastically binarize
        return img 
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class FixedBin(object):
    def __call__(self, img):
        assert(not img.max() > 1 and img.min() > 0) # valid input 
        img = (img > 0.5).float()  
        return img 
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

def dataset_attribute(dataset, attri):
    d2attr = {}
    d2attr['mnist'] = {
            'nclass': 10, 
            'out_dist': 'bernoulli',
            'pixel_class': 1,
            'imgd': 1,
            'canvsize': 28, 'imgshape': (28,28)
            }
    d2attr['stoch_mnist'] = deepcopy(d2attr['mnist'])
    d2attr['omni32l'] = {
            'nclass': 50,
            'out_dist': 'bernoulli', 
            'pixel_class': 1,
            'imgd':1,
            'canvsize': 28, 'imgshape': (52,52)
            }
    d2attr['omni32'] = {
            'nclass': 50,
            'out_dist': 'bernoulli', 
            'pixel_class': 1,
            'imgd':1,
            'canvsize': 28, 'imgshape': (28,28)
            }
    d2attr['omni'] = {
            'nclass': 50,
            'out_dist': 'bernoulli', 
            'pixel_class': 1,
            'imgd':1,
            'canvsize': 28, 'imgshape': (28,28)
            }
    d2attr['omnibr'] = deepcopy(d2attr['omni']) 
    d2attr['imgnet'] = {
            'nclass': 10, 
            'out_dist': 'cat',
            'pixel_class': 256,
            'imgd':3,
            'canvsize': 84, 'imgshape': (84,84,3)
            }

    d2attr['cifar'] = {
            'nclass': 10, 
            'out_dist': 'cat',
            'pixel_class': 256,
            'imgd':3,
            'canvsize': 32, 'imgshape': (32,32,3)
            }

    d2attr['cifarc'] = { # output space is continuous 
            'nclass': 10, 
            'out_dist': 'logistic',
            'pixel_class': 1,
            'imgd':3,
            'canvsize': 32, 'imgshape': (32,32,3)
            }
    # continuous output space, mixture_logistic 
    d2attr['cifarcm'] = deepcopy(d2attr['cifarc'])
    d2attr['cifarcm']['out_dist'] = 'mixture_logistic'

    d2attr['cifarc2s'] = deepcopy(d2attr['cifarc'])
    d2attr['cifarc2s']['out_dist'] = 'gaussian_as2svae'
    d2attr['cifarcs'] = deepcopy(d2attr['cifarc'])
    d2attr['cifarcs']['out_dist'] = 'gaussian'
    d2attr['cifarcl2'] = deepcopy(d2attr['cifarc'])
    d2attr['cifarcl2']['out_dist'] = 'l2' 

    d2attr['cifarg'] = { # gray scaled hidden 
            'nclass': 10, 
            'out_dist': 'cat',
            'pixel_class': 256,
            'imgd':1,
            'canvsize': 32, 'imgshape': (32,32)
            }

    d2attr['cifargs'] = deepcopy(d2attr['cifarg']) # gray scaled + sobel filtered hidden  

    ##d2attr['svhng'] = { 'nclass': 10, 'out_dist': 'cat', 'pixel_class': 256, 'imgd':1, 'canvsize': 32, 'imgshape': (32,32) }
    ##d2attr['svhngs'] = { 'nclass': 10, 'out_dist': 'cat', 'pixel_class': 256, 'imgd':1, 'canvsize': 32, 'imgshape': (32,32) }
    ##d2attr['svhn'] = { 'nclass': 10, 'out_dist': 'cat', 'pixel_class': 256, 'imgd':3, 'canvsize': 32, 'imgshape': (32,32,3) }
    ##d2attr['stl10g'] = { 'nclass': 10, 'out_dist': 'cat', 'pixel_class': 256, 'imgd':1, 'canvsize': 96, 'imgshape': (96,96) }
    ##d2attr['stl10gs'] = { 'nclass': 10, 'out_dist': 'cat', 'pixel_class': 256, 'imgd':1, 'canvsize': 96, 'imgshape': (96,96) }
    ##d2attr['stl10'] = { 'nclass': 10, 'out_dist': 'cat', 'pixel_class': 256, 'imgd':3, 'canvsize': 96, 'imgshape': (96,96,3) }
    # -----------------------------------------------------
    # celeba, canvsize=64, crop by 148x148 (follow realNVP)
    # -----------------------------------------------------
    ## softmax: 
    d2attr['celeba'] = { # 'nclass': 10, # 'pixel_class': 256,
        'out_dist': 'cat', 'imgd':3,
        'canvsize': 64, 'imgshape': (64,64,3), 'crop_size': 148}
    d2attr['celeba']['pixel_class'] = 256 
    ## continuous output space celebac
    d2attr['celebac'] = deepcopy(d2attr['celeba']) 
    d2attr['celebac']['out_dist'] = 'logistic'
    d2attr['celebac']['pixel_class'] = 1 

    d2attr['celebacs'] = deepcopy(d2attr['celebac'])
    d2attr['celebacs']['out_dist'] = 'gaussian'
    d2attr['celebacm'] = deepcopy(d2attr['celeba']) 
    d2attr['celebacm']['out_dist'] = 'mixture_logistic'
    d2attr['celebacm']['pixel_class'] = 1 
    ## use l2 loss, follow RAE paper 
    d2attr['celebacl2'] = deepcopy(d2attr['celebac']) 
    d2attr['celebacl2']['out_dist'] = 'l2'
    
    ## for canv only, single channel 
    d2attr['celebag'] = deepcopy(d2attr['celeba'])
    d2attr['celebags'] = deepcopy(d2attr['celeba'])
    d2attr['celebag']['imgshape'] = (64,64)
    d2attr['celebags']['imgshape'] = (64,64)
    d2attr['celebag']['imgd'] = 1 
    d2attr['celebags']['imgd'] = 1 
    
    # -----------------------------------------------------
    # celeba, canvsize=32(downsampled), crop by 148x148 (follow realNVP)
    # -----------------------------------------------------
    ## softmax 
    d2attr['celeba32'] = deepcopy(d2attr['celeba'])
    d2attr['celeba32'][  'canvsize'] = 32 
    ## logistic output space  
    d2attr['celebac32'] = deepcopy(d2attr['celebac'])
    d2attr['celebac32'][  'canvsize'] = 32 
    d2attr['celebac32i32'] = deepcopy(d2attr['celebac32']) 
    d2attr['celebac32i32']['imgshape'] = (32,32,3) 

    ## logistic output space  
    d2attr['celebacm32'] = deepcopy(d2attr['celebacm'])
    d2attr['celebacm32'][  'canvsize'] = 32 

    ## for canv only: 
    d2attr['celebag32'] = deepcopy(d2attr['celebag'])
    d2attr['celebags32'] = deepcopy(d2attr['celebags'])
    d2attr['celebag32'][ 'canvsize'] = 32 
    d2attr['celebags32']['canvsize'] = 32 
    ## del: only canvas_size is 32, imgshape is still 64 
    #d2attr['celeba32']['imgshape'] = (32,32,3)
    #d2attr['celebag32']['imgshape'] = (32,32)
    #d2attr['celebags32']['imgshape'] = (32,32)
    ## softmax: 
    ## # celeba dataset crop at 140x140, from celebaHQ  
    ## #TODO: change them to hq64_5bit's new name 
    ## d2attr['celeba14'] = deepcopy(d2attr['celeba']) 
    ## d2attr['celeba14c'] = deepcopy(d2attr['celebac']) 
    ## d2attr['celeba14g'] = deepcopy(d2attr['celebag']) 
    ## d2attr['celeba14gs'] = deepcopy(d2attr['celebags']) 
    ## d2attr['celeba14gs32'] = deepcopy(d2attr['celebags32']) 
    ## d2attr['celeba14g32'] = deepcopy(d2attr['celebag32']) 
    ## d2attr['celeba14c32'] = deepcopy(d2attr['celebac32']) 

    # celeba dataset crop at 140x140, from celeba  
    d2attr['celeba40'] = deepcopy(d2attr['celeba']) 
    d2attr['celeba40c'] = deepcopy(d2attr['celebac']) 
    d2attr['celeba40g'] = deepcopy(d2attr['celebag']) 
    d2attr['celeba40gs'] = deepcopy(d2attr['celebags']) 
    d2attr['celeba40gc'] = deepcopy(d2attr['celebags']) 

    d2attr['celeba40gs32'] = deepcopy(d2attr['celebags32']) 
    d2attr['celeba40g32'] = deepcopy(d2attr['celebag32']) 
    d2attr['celeba40c32'] = deepcopy(d2attr['celebac32']) 
    # set the crop size to be 140 
    for k in ['celeba40', 'celeba40c',   'celeba40g',   'celeba40gs', 'celeba40gc', 
                          'celeba40c32', 'celeba40g32', 'celeba40gs32' ]:
        d2attr[k]['crop_size'] = 140 

    # add canny edge 
    d2attr['celeba40gc32'] = deepcopy(d2attr['celeba40gs32']) 

    # celeba with cropped at center 140x140 
    d2attr['celebaS140'] = {'out_dist': 'logistic', 'imgd':3, 
            'canvsize': 140, 'imgshape': (140,140,3), 'crop_size': 140} 
    d2attr['celebaS140gs'] = {'out_dist': 'logistic', 'imgd':1, 
            'canvsize': 140, 'imgshape': (140,140  ), 'crop_size': 140} 
    d2attr['celebaS140gc'] = {'out_dist': 'logistic', 'imgd':1, 
            'canvsize': 140, 'imgshape': (140,140  ), 'crop_size': 140}

    d2attr['celebahq'] = {'out_dist': 'mixture_logistic', 'imgd':3,
            'canvsize': 256, 'imgshape': (256,256,3), 'pixel_class': 1}

    d2attr['celebahq128'] = {'out_dist': 'mixture_logistic', 'imgd':3,
            'canvsize': 128, 'imgshape': (256,256,3), 'pixel_class': 1}

    d2attr['celebahqc128'] = {'out_dist': 'logistic', 'imgd':3,
            'canvsize': 128, 'imgshape': (256,256,3), 'pixel_class': 1}

    d2attr['celebahqc64'] = {'out_dist': 'logistic', 'imgd':3,
            'canvsize': 64, 'imgshape': (128,128,3), 'pixel_class': 1}
    d2attr['celebahqc64i256'] = {'out_dist': 'logistic', 'imgd':3,
            'canvsize': 64, 'imgshape': (256,256,3), 'pixel_class': 1}
    d2attr['celebahqc32i64'] = {'out_dist': 'logistic', 'imgd':3,
            'canvsize': 32, 'imgshape': (64,64,3), 'pixel_class': 1}
    d2attr['celebahqc32i128'] = {'out_dist': 'logistic', 'imgd':3,
            'canvsize': 32, 'imgshape': (128,128,3), 'pixel_class': 1}
    d2attr['celebahqc128i128'] = {'out_dist': 'logistic', 'imgd':3,
            'canvsize': 128, 'imgshape': (128,128,3), 'pixel_class': 1}

    d2attr['celebahq_gs128'] = {'out_dist': 'mixture_logistic', 'imgd':1,
            'canvsize': 128, 'imgshape': (256,256), 'pixel_class': 1}

    d2attr['celebahq_g128'] = {'out_dist': 'mixture_logistic', 'imgd':1,
            'canvsize': 128, 'imgshape': (256,256), 'pixel_class': 1}
    if 'omnif' in dataset:
        output = d2attr['omni'][attri]
        return output
    elif 'mnistf' in dataset:
        return d2attr['stoch_mnist'][attri]
    elif re.search('cifar([\d])+', dataset): 
        matched = re.search('cifar([\d]+)', dataset).group(0)
        if dataset == matched: # return cifar  
            return d2attr['cifar'][attri]
        else:
            # remove the percent in the key 
            new_key = 'cifar'+dataset.split(matched)[-1] #
            return d2attr[new_key][attri]
    elif re.search('celebaf([\d]+)', dataset): 
        matched = re.search('celebaf([\d]+)', dataset).group(0)
        if dataset == matched: # return cifar  
            return d2attr['celeba'][attri]
        else:
            # remove the percent in the key 
            new_key = 'celeba'+dataset.split(matched)[-1] #
            return d2attr[new_key][attri]

    elif dataset not in d2attr:
        raise ValueError('unknown dataset: %s'%dataset)
    elif attri not in d2attr[dataset]:
        raise ValueError('unknown value %s for dataset: %s'%(attri, dataset))
    return d2attr[dataset][attri]

def split_val_from_train(dataset): 
    if 'omnif' in dataset:
        return 0 
    elif re.search('cifar[0-9][0-9]', dataset): 
        return 0
    elif re.search('celebaf([\d]+)', dataset): 
        return 0
    elif re.search('mnistf([\d]+)', dataset): 
        return 0
    else:
        return 1

def get_nclass(dataset): 
    return dataset_attribute(dataset, 'nclass') 

def get_out_dist_npara(dist, k=1): # dataset): 
    # number of parameters for the distributions 
    if dist in ['bernoulli', 'l2']:
        return 1 
    elif dist == 'gaussian_as2svae':
        return 1 # gamma is treated as a model's parameters 
    elif dist in ['gaussian']:
        assert(k==1), 'non mixture mode, require k=1'
        return 2 
    elif dist == 'logistic':
        assert(k==1), 'non mixture mode, require k=1, get k=%d'%k
        return 2 
    elif dist in ['mixlogistic', 'mixture_logistic']:
        return 3*k
        ## return 2 * k 
        # k: number of mixture 
    elif dist == 'cat':
        return k
    else:
        raise ValueError('unknown distribution: %s'%dist)

def get_pixel_nclass(dataset): 
    return dataset_attribute(dataset, 'pixel_class') 

def get_out_dist(dataset): 
    return dataset_attribute(dataset, 'out_dist') 

def get_imgd(dataset): 
    return dataset_attribute(dataset, 'imgd') 

def get_imgsize(dataset): 
    return get_imgshape(dataset)[0]

def get_canvsize(dataset): 
    return dataset_attribute(dataset, 'canvsize') 

def get_cropsize(dataset): 
    crop_size = dataset_attribute(dataset, 'crop_size') 
    # logger.info('dataset: {}, crop_size={}', dataset, crop_size)
    assert(crop_size is not None), 'not get crop_size for %s'%dataset 
    return crop_size 

def get_imgshape(dataset): 
    return dataset_attribute(dataset, 'imgshape') 

def get_canvshape(dataset): 
    imgshape = dataset_attribute(dataset, 'imgshape') 
    canvsize = dataset_attribute(dataset, 'canvsize') 
    if len(imgshape) == 3: # 3 dim 
        assert(imgshape[-1] == 3)
        return (canvsize,canvsize,3)
    else:
        assert(len(imgshape) == 2)
        return (canvsize,canvsize,1)

def get_nfloc(dataset, ww):
    # num of fixed location 
    imgh = get_canvsize(dataset) 
    nloc = imgh // ww + 1 if imgh % ww > 1 else imgh // ww 
    return nloc 

def get_floc_pad(dataset, ww):
    # num of pixel for padding 
    nloc = get_nfloc(dataset, ww)
    imgh = get_canvsize(dataset) 
    pad_size = (nloc) * ww - imgh 
    leftp = pad_size // 2 
    rightp = pad_size - leftp 
    return leftp, rightp 

def get_canvasd(cfg):
    imgd = get_imgd(cfg.dataset)
    if cfg.cat_vae.canvas_dim > 0: 
        canvasd = cfg.cat_vae.canvas_dim
    else: 
        canvasd = imgd 
    return canvasd 

