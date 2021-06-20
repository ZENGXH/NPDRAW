# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque
# from tensorboardX import SummaryWriter
import numpy as np
import os
import pickle
import torch
import time
import atexit
from PIL import Image
# from tabulate import tabulate
# import prettytable 
from loguru import logger 
global outname, fields, tags, summary 
outname = '/tmp/random_exp.pkl' 
fields  = [] 
tags = []
summary = {}
DEBUG=1
def flatten_dict(dd, separator ='_', prefix =''): 
        return { prefix + separator + k if prefix else k : v for kk, vv in dd.items() \
                for k, v in flatten_dict(vv, separator, kk).items() } if isinstance(dd, dict) else { prefix : dd } 

def create_exp_dir(exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0
        self.this_epoch = 0 

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        self.this_epoch += 1

    def clear_epo_counter(self):
        self.this_epoch = 0 # go back

    @property
    def mean_of_this_epoch(self): # , window_size):
        return self.mean_of_last(self.this_epoch+1)

    def mean_of_last(self, window_size):
        if window_size >= self.count:
            return self.total / self.count 
        else:
            total = self.series[-window_size:]
            m = sum(total) / len(total)
            return m

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()
    @property
    def global_avg(self):
        return self.total / self.count

class MetricLogger(object):
    def __init__(self, delimiter=" ", exp_dir=None, save_interval=1, active=0):
        r'''
        args:
            save_interval, save every x hours 

        '''
        self.active = active
        self.meters = defaultdict(SmoothedValue)
        self.meters_epochwise = defaultdict(SmoothedValue)

        self.delimiter = delimiter # printing 
        self.epo = 1

        # Comet support 
        self.cmt = None
        self.exp = None
        # tf board support 
        self.tf_dir = None 
        self.tfb = None
        # argsments 
        self.cfg = None 

        self.tags = []

        # denote train and test 
        self.prefix = ''

        # hyperparameters 
        self.stats = {'flat_parameters': {}} 
        # for experiment id 
        self.exp_start_time = time.strftime('%Y-%b-%d-%H-%M-%S')

        # used for periodic saving 
        self.init_tic = time.time()
        self.save_toc = 60*60*save_interval # save per x hours 
        self.interval_id = -1

        if exp_dir is None: 
            self.exp_dir = '/tmp/'
        else:
            self.exp_dir = exp_dir 
        self.outname = os.path.join(self.exp_dir, self.exp_start_time, 'train_metric.p')
        self.model_str = ''
        # fields = self.fields()

        ## recording best score and epo 
        self.exp_score = None 
        self.best_epo = 0 
        self.epoch = 0 

    def log_model(self, model): #, input):
        self.model_str = '{}'.format(model)
        if self.exp is not None:
            # logger.info('LOG MODEL')
            self.exp.set_model_graph(model, overwrite=True)

    def log_made_grid(self, name, grid, i=0):
        '''
        Args: 
            name (str): 
            img (tensor): [????]
        '''
        if self.exp is not None:
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr) 
            self.exp.log_image(im, name, step=i)
            return im 
        else:
            return 0 


    def log_pil(self, name, img, i=-1):
        if i == -1:
            i = self.epoch 
        if self.exp is not None: 
            self.exp.log_image(img, name, step=i) 
            return 1
        return 0

    def log_image(self, name, img, i=0):
        '''
        Args: 
            name (str): 
            img (tensor): [????]
        '''
        #if self.tfb is not None and type(img) != str:
        #    self.tfb.add_image(name, img, i) 
        if self.exp is not None:
            if type(img) is str:
                self.exp.log_image(img, name, step=i)
                return 1 #im 
            elif img.max() < 100: # [0-1]
                ndarr = img.mul(255).add_(0.5).clamp_(0, 255).to(
                        'cpu').squeeze().numpy().astype(np.uint8)
                # print(ndarr.shape) 
                im = Image.fromarray(ndarr[0].reshape(-1, ndarr.shape[-1]))
                self.exp.log_image(im, name, step=i)
                return 1 #im 
            else:
                im = img.to('cpu').squeeze().numpy()[0]
                self.exp.log_image(im, name, step=i)
                return 1 #im 
        else:
            return 0 # img 
    def log_file(self, name):
        if self.exp is not None:
            if 'png' in name:
                self.exp.log_image(name)
            else:
                self.exp.log_asset(name)


    @property 
    def hyperparameters(self):
        return self.stats['flat_parameters'] if 'flat_parameters' in self.stats else {}

    # add hyperparameters
    def log_hp(self, args_dict):
        # log hyper parameters 
        self.cfg = args_dict
        flat_para = flatten_dict(args_dict, separator='-')
        self.stats['flat_parameters'] = flat_para
        self.comet_hp()
        if hasattr(self.cfg, 'cmt') and len(self.cfg.cmt) > 0:
            self.cmt = self.cfg.cmt 

    def log_text(self, t, tag=''):
        if self.tfb is not None:
            self.tfb.add_text(t, tag)
        # logger.info('{}: {}'.format(tag, t))

    # add tags 
    def add_tags(self, tlist):
        # add list of string 
        self.tags.extend(tlist)
        global tags
        tags = self.tags
        self.comet_tag()

    def add_tag(self, t):
        # add string 
        self.tags.append(t)
        global tags 
        tags = self.tags
        self.comet_tag()

    def log_best(self, new_score, epo, trend='up'):
        if trend == 'up':
            if self.exp_score is not None and new_score < self.exp_score: return 
        elif trend == 'down':
            if self.exp_score is not None and new_score > self.exp_score: return 
        else:
            raise ValueError(trend)

        self.exp_score = new_score 
        self.best_epo = epo 
        self.exp.log_metrics({'best_score': new_score})
        self.exp.log_metrics({'best_epo':   epo})

    # staus 
    def train(self):
        self.prefix = 'train_'
        self.stage = 'train'
        
    def eval(self):
        self.prefix = 'eval_'
        self.stage = 'eval'
    
    def test(self):
        self.prefix = 'test_'
        self.stage = 'test'

    def start_epoch(self, epo=-1): 
        # logger.info('logger start epoch {}|{}', epo, self.epoch) 
        if epo == 0 and self.epoch == 0: 
            return # do nothing
        if epo == self.epoch: 
            return 1 # do nothing 

        if epo > -1: 
            self.epoch = epo 

        for name, meter in self.meters.items():
            v = meter.mean_of_this_epoch # get the avg of current epoch, and then clear the counter 
            meter.clear_epo_counter()
            epo_name = 'epo_' + name
            # self.meters_epochwise[epo_name].update(v)
            if self.exp is not None:
                self.exp.log_metrics({epo_name:v})
                # logger.info('epoch {}; {}={:.3f}', epo-1, epo_name, v)
            if self.tfb is not None:
                self.tfb.add_scalar(epo_name, v, self.epoch)

    # TF board support 
    def reg_tfboard(self, tf_dir):
        if self.tfb is not None:
            raise RuntimeWarning('my tf board is registered, skip the reg now; check %s'%self.tf_dir)
            return 
        self.tf_dir = tf_dir 
        logger.info('register tf writer at %s'%tf_dir)
        self.tfb = SummaryWriter(tf_dir)
        self.outname = os.path.join(self.tf_dir, self.exp_start_time, 'train_metric.p')

    # COMET experiment 
    def reg_exp(self, exp):
        if self.exp is not None:
            raise RuntimeWarning('reg_exp is called, but exp already exists')
        else:
            self.exp = exp 
            logger.info('get exp at {}', self.exp.url)
            self.comet_hp()

    def comet_hp(self):
        # log config to comet 
        if self.exp is None:
            return 
        if 'flat_parameters' in self.stats.keys():
            self.exp.log_parameters(self.stats['flat_parameters'])

    def comet_tag(self):
        if self.exp is None:
            return 
        self.exp.add_tags(self.tags)

    def log_html(self, msg):
        if self.exp is None:
            return None
        else:
            msg = msg.split('\n')
            for m in msg:
                self.exp.log_html( m + ' <br> ')
    @property
    def comet_url(self):
        if self.exp is None:
            return '' 
        else:
            return self.exp.url
    #def update_int(self, **kwargs):
    #    self.save_periodic()
    #    for k, v in kwargs.items():
    #        if isinstance(v, torch.Tensor):
    #            v = v.item()
    #        assert isinstance(v, (float, int))
    #        self.meters_int[self.prefix + k].update(v)
    #        if self.exp is not None:
    #            self.exp.log_metrics({k:v}, prefix=self.prefix)
    #        if self.tfb is not None:
    #            self.tfb.add_scalar(self.prefix + k, v)

    # update, clear and close + save 
    def update(self, **kwargs):
        """
        save to both the exp-comet object if exist 
        and tensorboardX is exist 
        """
        self.save_periodic()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > 1:
                    logger.info('get element with numel : {}; k {}; do avg', v.numel(), k)
                    v = v.mean()
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[self.prefix + k].update(v)
            #if self.exp is not None:
            #    self.exp.log_metrics({k:v}, prefix=self.prefix.strip('_'))
            if self.tfb is not None:
                self.tfb.add_scalar(self.prefix + k, v, 
                        self.meters[self.prefix+k].count)

    def set_exp_dir(self, exp_dir):
        if self.exp is not None:
            self.exp.log_other('exp', exp_dir) 
            self.exp.log_other('url', self.comet_url) 
            if self.cmt is not None:
                self.exp.log_other('cmt', self.cmt) # comet_url) 
    
    def write(self, key, val):
        if self.exp is not None:
            self.exp.log_other(key, val)

    def clear(self):
        self.meters = defaultdict(SmoothedValue) 

    # save:
    def close(self):
        if self.tfb is not None:
            self.tfb.close() 
            logger.info('close tf board at %s'%self.tf_dir)
        self.save()

    def save_periodic(self):
        if not self.active or DEBUG: return 
        cur_time = time.time() 
        exp_time = cur_time - self.init_tic 
        interval_id = exp_time // self.save_toc 
        if interval_id > self.interval_id:
            logger.info('metric periodic %d - %d saved at %s'%(interval_id, self.interval_id, self.outname))
            self.interval_id = interval_id
            self.save()
            # fields = self.fields() 
            self.write_summary()     
            #import csv
            #report = self.write_summary()
            #with open('exp.csv', 'a') as csvFile:
            #    fields = report.keys()
            #    writer = csv.DictWriter(csvFile, fieldnames=fields)
            #    writer.writeheader()
            #    writer.writerow(report)
            #    print("writing completed")
            #    csvFile.close()

    def save(self):
        if not self.active or DEBUG: return 
        create_exp_dir(os.path.dirname(self.outname))
        pickle.dump(self.fields(), open(self.outname, 'wb'))
 
    #@atexit.register
    #def save_exit():
    #    if DEBUG:
    #        print('debugging, skip save')
    #        return 
    #    create_exp_dir(os.path.dirname(outname))
    #    pickle.dump(fields, open(outname, 'wb'))
    #    print('I saved log at %s !'%outname)
    #    for t in tags:
    #        print('tags: #%s'%t)
    #    import csv
    #    report = summary # self.write_summary()
    #    with open('exp.csv', 'a') as csvFile:
    #        fields = report.keys()
    #        writer = csv.DictWriter(csvFile, fieldnames=fields)
    #        writer.writeheader()
    #        writer.writerow(report)
    #        print("writing completed")
    #        csvFile.close()

    #    # headers = list(summary.keys())
    #    #headers = []
    #    #values = []
    #    #for k, v in summary.items():
    #    #    headers.append(k)
    #    #    values.append(v) 

    #    #if len(summary) > 0:
    #    #    info = '\n{}\n'.format(tabulate([values], headers, tablefmt="github"))
    #    #    f = open('EXPERIMENT.md', 'a')
    #    #    f.write(info)
    #    #    f.close()


    def write_summary(self):
        tablelist = defaultdict()
        if self.exp is not None:
            tablelist['comet_key'] = self.exp.get_key()
        for k, v in self.hyperparameters.items():
            tablelist['cfg-%s'%k] = v

        try:
            tablelist['model_name'] = self.cfg.model.name
            tablelist['runner'] = self.cfg.runner 
            tablelist['dataset'] = self.cfg.dataset.loader_name
        except:
            pass

        tablelist['start_time'] = self.exp_start_time 

        tablelist['exp_dir'] = self.exp_dir 
        tablelist['run_time'] = (time.time() - self.init_tic)/3600.0

        for k, v in self.meters.items():
            tablelist['metric-ITERS-%s'%k] = v.global_avg
            tablelist['metric-EPO-%s'%k] = vmean_of_this_epoch 
        tablelist['best_epo'] = self.best_epo
        tablelist['best_score'] = self.exp_score
        global summary 
        summary = tablelist
        return tablelist 

    def fields(self):
        return [self.meters, self.stats, self.tags, self.cfg, self.model_str]

    # init 
    # TODO: @static_method 
    def load_from_pkl(self, k):
        pkl = pickle.load(open(k, 'rb'))
        self.meters, self.stats, self.tags = pkl

    #def __getattr__(self, attr):
    #    if attr in self.meters:
    #        return self.meters[attr]
    #    if attr in self.__dict__:
    #        return self.__dict__[attr]
    #    raise AttributeError("'{}' object has no attribute '{}'".format(
    #                type(self).__name__, attr))

    def msg(self, stage='', niters=-1):
        names = []
        if stage == '':
            stage = self.stage 
        if stage == 'train' or stage == 'eval' or stage == 'test':
            for name in self.meters.keys():
                if name[:len(stage)] == stage:
                    names.append(name) 
        #elif stage == 'all':
        #    names = list(self.meters.keys())
        #else:
        #    raise ValueError

        loss_str = []
        loss_str.append('[{:<5}] E{:<3}'.format(stage, self.epoch))
        if niters > -1: 
            loss_str.append('i{0:<3}'.format(niters))
        #else: 
        #    loss_str.append('i%d'%meter.this_epoch)
        names = sorted(names) 
        for name in names: 
            meter = self.meters[name]
            if niters > -1: 
                avg_val = meter.mean_of_last(niters)
            else:
                avg_val = meter.mean_of_this_epoch 
            name = name.split(stage+'_')[-1]
            if avg_val == 0:
                loss_str.append("|{}: 0".format(name)) 
            elif avg_val < 0.01:
                loss_str.append("|{}: {:6.2e}".format(name, avg_val).replace('e-0', 'e-').replace('.00e','e'))
            elif avg_val < 100:
                loss_str.append("|{}: {:4.2f}".format(name, avg_val))
            elif avg_val < 10:
                loss_str.append("|{}: {:3.2f}".format(name, avg_val))
            else:
                loss_str.append( "|{}: {:6.2f}".format(name, avg_val))
        return self.delimiter.join(loss_str)
    # output 
    def __str__(self):
        loss_str = []
        loss_str.append('E%d'%self.epoch)
        #if len(self.tags) > 0:
        #    loss_str.append('T%s'%'#'.join(self.tags))
        if self.exp_score is not None:
            loss_str.append('Best %.2f@%d'%(self.exp_score, self.best_epo))
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:6.3f}".format(name, meter.mean_of_this_epoch)
                # "{}: {:.3f} ({:.3f})".format(name, meter.mean_of_this_epoch, meter.global_avg)
            )

        return self.delimiter.join(loss_str)
