#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:05 2019

@author: cxue2
"""
import sys
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from tcn import TCN
from data import collate_fn
from misc import calc_performance_metrics, show_performance_metrics

class Model_CNN:
    """
    CNN model;
    """
    def __init__(self, n_concat, device='cpu'):
        """
        init method;
        """
        self.n_concat = n_concat
        self.nn = TCN(device)
        self.to(device)
        self.device = device

    def fit(self, dset_trn, dset_vld=None, n_epoch=32, b_size=4, lr=.001, weights=None):
        """
        fit method;
        """
        if weights is None:
            weights = []
        wrs = WeightedRandomSampler(dset_trn.df_sampling_weights,
            int(np.ceil(len(dset_trn) * (2/3))), replacement=False)
        # initialize data loaders
        kwargs = {'batch_size': b_size,
                  # 'shuffle': True,
                  'shuffle': False,
                  'num_workers': 1,
                  'collate_fn': collate_fn,
                  'sampler': wrs
                  }
        dldr_trn = data.DataLoader(dset_trn, **kwargs)
        # initialize loss function and optimizer
        weights = torch.FloatTensor(weights).cuda(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        # for model selection
        vld_mcc = -1
        for epoch in range(n_epoch):
            # set model to training mode
            self.nn.train()
            # model performance statistics
            cum_loss, cum_corr, count = 0, 0, 0
            # training loop
            with tqdm(total=len(dset_trn), desc='Epoch {:03d} (TRN)'.format(epoch),
                ascii=True, bar_format='{l_bar}{r_bar}', file=sys.stdout) as pbar:
                for Xs, ys, _ in dldr_trn:
                    # mount data to device
                    for _, xs_item in enumerate(Xs):
                        xs_item = torch.tensor(xs_item, dtype=torch.float32, device=self.nn.device)
                        xs_item = xs_item.permute(1, 0)
                        xs_item = xs_item.view(1, xs_item.shape[0], xs_item.shape[1])
                    ys = torch.tensor(ys, dtype=torch.long, device=self.nn.device)
                    # forward and backward propagation
                    self.nn.zero_grad()
                    scores = self.nn(Xs)
                    # loss
                    loss = loss_fn(scores, ys)
                    loss.backward()
                    op.step()
                    pred = torch.argmax(scores, 1)
                    # accumulated loss
                    cum_loss += loss.data.cpu().numpy() * len(ys)
                    # accumulated no. of correct predictions
                    cum_corr += (pred == ys).sum().data.cpu().numpy()
                    # accumulated no. of processed samples
                    count += len(ys)
                    # update statistics and progress bar
                    pbar.set_postfix({
                        'loss': '{:.6f}'.format(cum_loss / count),
                        'acc' : '{:.6f}'.format(cum_corr / count)
                    })
                    pbar.update(len(ys))
            # forward validation dataset
            scr = self.prob(dset_vld)
            # calculate audio-level performance metrics
            met = calc_performance_metrics(scr, dset_vld.labels)
            print('Audio-level validation performance:')
            show_performance_metrics(met)
            print()
            # save model
            if np.isnan(met['mcc']):
                continue
            if vld_mcc <= met['mcc']:
                vld_mcc = met['mcc']
                self.save_model('./tmp.pt')
        # load best model
        if dset_vld is not None and vld_mcc != -1:
            self.load_model('./tmp.pt')

    def eval(self, dset, b_size=32):
        """
        eval model;
        """
        # set model to validation mode
        self.nn.eval()
        # initialize data loader
        kwargs = {'batch_size': b_size,
                  'shuffle': False,
                  'num_workers': 1,
                  'collate_fn': collate_fn}
        dldr = data.DataLoader(dset, **kwargs)
        # list to store result (i.e. all outputs)
        rsl = []
        # evaluation loop
        with torch.set_grad_enabled(False):
            with tqdm(total=len(dset), desc='Epoch ___ (EVL)', ascii=True,
                bar_format='{l_bar}{r_bar}', file=sys.stdout) as pbar:
                for Xs, _, _ in dldr:
                    # mount data to device
                    for _, xs_item in enumerate(Xs):
                        xs_item = torch.tensor(xs_item, dtype=torch.float32, device=self.nn.device)
                        xs_item = xs_item.permute(1, 0)
                        xs_item = xs_item.view(1, xs_item.shape[0], xs_item.shape[1])
                    out = self.nn(Xs)
                    # append batch outputs to result
                    rsl.append(out.data.cpu().numpy())
                    # progress bar
                    pbar.update(len(Xs))
        # concatenate all batch outputs
        rsl = np.concatenate(rsl)
        return rsl

    def prob(self, dset, b_size=32):
        """
        calculate model output (probability);
        """
        # get network output
        rsl = self.eval(dset, b_size)
        # convert output to probability by softmax
        rsl = np.exp(rsl)[:,1] / np.sum(np.exp(rsl), axis=1)
        return rsl

    def save_model(self, fp):
        """
        save model as a file;
        """
        torch.save(self.nn, fp)

    def load_model(self, fp):
        """
        load a model file;
        """
        self.nn = torch.load(fp)

    def to(self, device):
        """
        call model.to() and set device;
        check range of gpu devices (currently have 4 gpus);
        """
        assert device in ['cpu', 0, 1, 2, 3], 'Invalid device.'
        self.nn.to(device)
        self.nn.device = device

    def eval_wo_gpool(self, dset, b_size=32):
        """
        model eval() without gpool;
        """
        # set model to validation mode
        self.nn.eval()
        # initialize data loader
        kwargs = {'batch_size': b_size,
                  'shuffle': False,
                  'num_workers': 1,
                  'collate_fn': collate_fn}
        dldr = data.DataLoader(dset, **kwargs)
        # list to store result (i.e. all outputs)
        rsl = []
        # evaluation loop
        with torch.set_grad_enabled(False):
            with tqdm(total=len(dset), desc='Epoch ___ (EVL)', ascii=True,
                bar_format='{l_bar}{r_bar}', file=sys.stdout) as pbar:
                for Xs, _, _ in dldr:
                    # mount data to device
                    for _, xs_item in enumerate(Xs):
                        xs_item = torch.tensor(xs_item, dtype=torch.float32, device=self.nn.device)
                        xs_item = xs_item.permute(1, 0)
                        xs_item = xs_item.view(1, xs_item.shape[0], xs_item.shape[1])
                    out = self.nn.forward_wo_gpool(Xs)
                    # append batch outputs to result
                    rsl += out
                    # progress bar
                    pbar.update(len(Xs))
        return rsl
