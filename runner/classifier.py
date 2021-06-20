import os
from comet_ml import Experiment
import torch 
import time
from tqdm import tqdm
import numpy as np
from loguru import logger
from runner.vae_runner import Runner 
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import sklearn.cluster
import torchvision 
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
COMPUTE_ELBO=1
EVAL_CLS=0
# --------------------
# compute matched cls acc 
# --------------------
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    y: true labels, numpy.array with shape `(n_samples,)`
    y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
    accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1 # number of classes 
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    indi, indj = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(indi, indj)]) * 1.0 / y_pred.size
# --------------------
# Plotting helpers
# --------------------
FAST=1
def plot_tsne(avg_q_z_x, labels, output_dir): # , test_loader, args):
    classes = torch.unique(labels, sorted=True).numpy()

    #p_x_z, avg_q_z_x = model(data)
    tsne = TSNE(n_components=2, random_state=0)
    z_embed = tsne.fit_transform(avg_q_z_x.cpu().numpy())  # map the posterior mean

    fig = plt.figure()
    for i in classes:
        mask = labels.cpu().numpy() == i
        plt.scatter(z_embed[mask, 0], z_embed[mask, 1], s=10, label=str(i))

    plt.title('Latent variable T-SNE embedding per class')
    plt.legend()
    plt.gca().axis('off')
    fig.savefig(os.path.join(output_dir, 'tsne_embedding.png'))
#
#def bipartite_match(pred, gt, n_classes=None, presence=None):
#  """Does maximum biprartite matching between `pred` and `gt`."""
#
#  if n_classes is not None:
#    n_gt_labels, n_pred_labels = n_classes, n_classes
#  else:
#    n_gt_labels = np.unique(gt).shape[0]
#    n_pred_labels = np.unique(pred).shape[0]
#
#  cost_matrix = np.zeros([n_gt_labels, n_pred_labels], dtype=np.int32)
#  for label in range(n_gt_labels):
#    label_idx = (gt == label)
#    for new_label in range(n_pred_labels):
#      errors = np.equal(pred[label_idx], new_label).astype(np.float32)
#      if presence is not None:
#        errors *= presence[label_idx]
#
#      num_errors = errors.sum()
#      cost_matrix[label, new_label] = -num_errors
#
#  row_idx, col_idx = linear_sum_assignment(cost_matrix)
#  num_correct = -cost_matrix[row_idx, col_idx].sum()
#  acc = float(num_correct) / gt.shape[0]
#  from monty.collections import AttrDict
#  return AttrDict(assingment=(row_idx, col_idx), acc=acc,
#                  num_correct=num_correct)
#
#
#def cluster_classify(features, gt_label, n_classes, kmeans=None, max_iter=100):
#  """Performs clustering and evaluates it with bipartitate graph matching."""
#  if kmeans is None:
#    kmeans = sklearn.cluster.KMeans(
#        n_clusters=n_classes,
#        precompute_distances=True,
#        n_jobs=-1,
#        max_iter=max_iter,
#    )
#
#  kmeans = kmeans.fit(features)
#  pred_label = kmeans.predict(features)
#  return np.float32(bipartite_match(pred_label, gt_label, n_classes).acc)



class classifier_tsne(Runner):
    def __init__(self, cfg, metric, local_rank, sample_only=False):
        super().__init__(cfg, metric, local_rank, sample_only)
        # plot_tsne(model, test_loader, args)

    #@torch.no_grad()
    #def compute_elbo_nsample(self, epoch):
    #    torch.manual_seed(epoch) # freeze same random seed for testing 
    #    self.model.eval()
    #    self.metric.eval() 
    #    test_loss = []
    #    test_bce, test_kld = [], []
    #    avg_q_z_x, target = [], []
    #    tic = time.time() 
    #    num_sample = 50 
    #    img_size = self.img_size 
    #    output_dir = os.path.join(self.cfg.exp_dir, self.cfg.exp_name) 
    #    for i, (data, labels) in enumerate(self.test_loader):
    #        Bs = len(data)
    #        data = data.to(self.device).float() #.view(1,B,784)

    #        output, loss_dict = self.model.test_loss(data, num_sample=num_sample)
    #        self.metric.update(**loss_dict) #NLL=loss_dict['NLL'])
    #        if i % 1000 == 0:
    #            logger.info('num_sample: {} {}', 
    #                    num_sample, self.metric.msg('eval', i+1)) 
    #    msg = '[Eval] | E{} |num_sample: {} {} | {} | '.format(self.init_epoch-1, num_sample,
    #            self.metric.msg('eval', len(self.test_loader)), output_dir)
    #    logger.info('\n'+msg)
    #    with open('.results/classifier.md', 'a') as f:
    #        f.write(datetime.now().strftime('%m-%d-%H-%M-%S') + ' ') 
    #        f.write(msg+'\n')

    #    logger.info('forward test set time: %.3fs'%(time.time() - tic))

    @torch.no_grad()
    def test(self, epoch, num_sample):
        self.model.temp_gt_q = 0.0
        self.vis_recont(-1)
        #if hasattr(self.model, 'cls') and 'omni' not in self.cfg.dataset and EVAL_CLS:
        #    cls_feat = [self.model.cls(data.view(-1,self.input_dim).to(self.device)).cpu().numpy() \
        #            for data,_ in self.test_loader]
        #    cls_tar =  [tar.cpu().numpy() for _,tar in self.test_loader] 
        #    cls_feat = np.concatenate(cls_feat) 
        #    cls_tar = np.concatenate(cls_tar)
        #    out = self.compute_kmean_acc(cls_feat, cls_tar) #-1)
        #    AAE_acc = self.compute_AAE_acc(cls_feat, cls_tar) 
        #    msg = ''
        #    if AAE_acc > 0: 
        #        msg += 'AAE-acc: {:.3f}'.format(AAE_acc)
        #    msg += 'kmeans-acc: {:.3f}, nmi: {:.3f}'.format(out['acc'], out['nmi'])
        #    logger.info(msg) 
        if COMPUTE_ELBO:
            return self.compute_elbo_nsample(epoch, num_sample=num_sample) 

        #if self.cfg.eval.plot_tsne:
        #    torch.manual_seed(epoch) # freeze same random seed for testing 
        #    self.model.eval()
        #    test_loss = []
        #    test_bce, test_kld = [], []
        #    avg_q_z_x, target = [], []
        #    tic = time.time()
        #    for i, (data, labels) in enumerate(self.test_loader):
        #        data = data.to(self.device).float()
        #        Bs = len(data) 
        #        # output, loss_dict = self.model(data)
        #        output = {}
        #        output['cls'] = self.model.cls(data).view(Bs,-1)
        #        avg_q_z_x.append(output['cls'].cpu()) 
        #        target.append(labels.cpu()) 
        #    logger.info('forward test set time: %.3fs'%(time.time() - tic))
        #    output_dir = os.path.join(self.cfg.exp_dir, self.cfg.exp_name) 
        #    avg_q_z_x, target = torch.cat(avg_q_z_x), torch.cat(target) 
        #    plot_tsne(avg_q_z_x, target, output_dir)
        ##if self.cfg.eval.compute_cluster_acc:
        ##    self.eval_cls_acc(avg_q_z_x.numpy(), target.numpy())

#    @torch.no_grad() 
#    def eval_cls_acc(self, val_feat, val_label):
#        # Unsupervised classification via clustering
#        logger.info('Bipartite matching classification accuracy:')
#        avg_q_z_x, target = [], []
#        tic = time.time()
#        train_feat = val_feat 
#        train_label = val_label 
#        #for i, (data, labels) in enumerate(self.train_loader):
#        #    data = data.to(self.device).float()
#        #    Bs = len(data) 
#        #    # output, loss_dict = self.model(data)
#        #    output = {}
#        #    output['cls'] = self.model.cls(data).view(Bs,-1)
#        #    avg_q_z_x.append(output['cls'].cpu()) 
#        #    target.append(labels.cpu()) 
#        #logger.info('forward train set time: %.3fs'%(time.time() - tic))
#        #train_feat = torch.cat(avg_q_z_x).numpy() 
#        #train_label = torch.cat(target).numpy() 
#        tic = time.time()
#        logger.info('Start Clustering')
#        if FAST: 
#            from sklearn.cluster import MiniBatchKMeans # as KMeans
#            kmeans = MiniBatchKMeans(n_clusters=10, random_state=0).fit(train_feat)
#        else:
#            kmeans = sklearn.cluster.KMeans(
#                n_clusters=10,
#                precompute_distances=True,
#                n_jobs=-1,
#                max_iter=1000,
#                ).fit(train_feat)
#        logger.info('clustering time: %.3fs'%(time.time() - tic))
#        
#        train_acc = cluster_classify(train_feat, train_label, 10, 
#                                     kmeans)
#        logger.info('train acc: {:.3f}', train_acc)
#        y_pred = kmeans.fit_predict(val_feat) # eval) 
#        valid_acc = acc(val_label, y_pred) 
#        # valid_acc = cluster_classify(val_feat,   val_label, 10, 
#        #                             kmeans)
#        logger.info('valid acc: {:.3f}', valid_acc)
#        output_dir = os.path.join(self.cfg.exp_dir, self.cfg.exp_name) 
#        msg = '| {} | train_acc {:.4f} | valid_acc {:.4f} |\n'.format(output_dir, 
#                train_acc, valid_acc)
#        logger.info(msg)
#        with open('.results/record_cls.md', 'a') as f:
#            f.write(msg)
#        
    @torch.no_grad() 
    def sample_intera(self, epoch): 
        out = []
        N = 64 
        logger.info('start sampling, N={}', N)
        self.model.sample_10k = True # temp solution to turn off z&e controller 
        exp = Experiment()
        logger.info('exp url: {}', exp.url)
        for k in tqdm(range(10000 // N + 1)):
            # 10000 / 64 
            if self.cfg.model_name in ['cvae', 'cvae2', 'cvaer']:
                raise NotImplementedError
                #hid=torch.randn(64, *self.model.latent_shape) 
                #sample = self.model.sample(hid.to(self.device)) 
            elif self.cfg.model_name == 'vae':
                raise NotImplementedError
                #hid = torch.randn(N, self.model.hid_size)
                #sample = self.model.sample(hid.to(self.device)) 
            else:
                sample = self.model.sample_intera() 
                hid = None

            if type(sample) is tuple: 
                if self.cfg.model_name == 'pvae' or self.cfg.use_patch_bank:
                    sample, hid = sample
                    # hid = hid.cpu()
                else:
                    sample, _ = sample
            B = sample.shape[0]
            sample = sample.cpu().view(B, self.imgd, self.img_size, self.img_size)
            out.append(sample) 
            if True: #k < 10:
                filename = '%s/%s/'%(self.cfg.exp_dir, self.cfg.exp_name) + \
                        'sample_rand' + str(epoch) + '-%d.png'%k
                if hid is not None and hid.shape[-1] == self.img_size: 
                    hid = hid.cpu()
                    sample = torch.cat([sample, hid])
                fig = save_image(sample, filename, normalize=True, scale_each=True, nrow=B, pad_value=0.5)    
                # logger.info('save as {}', filename)
                exp.log_image(filename, 'sample', step=k) 
                #exit()
                ## logger.info('sample: {}; min={}; max={}', sample.shape, sample.min(), sample.max())
        out_pt = torch.cat(out) 
        logger.info('get output: {}', out_pt.shape) 
        # assert(out_pt.shape[0] > 10000), 'get output less than 10k sample'

        out = out_pt.numpy() 
        filename = '%s/%s/'%(self.cfg.exp_dir, 
                self.cfg.exp_name) + 'rand_sample_' + str(epoch) + '.npy'
        np.save(filename, out)
        if self.img_size > 28 and 'omni' in self.cfg.dataset:
            out_down = F.interpolate(out_pt, (28,28), mode='bilinear').numpy() 
            np.save(filename.replace('.npy', '_28.npy'), out_down)
        logger.info('save at %s'%filename)



