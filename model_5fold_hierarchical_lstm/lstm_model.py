#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:05 2019

@author: cxue2
"""

import sys
from lstm_bi import LSTM
#from lstm_stacked import LSTM
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
from tqdm import tqdm
from data import collate_fn, reshape_
from torch.utils import data
from collections import defaultdict
from misc import calc_performance_metrics, show_performance_metrics
from mfcc import MFCC_Extractor

class Model_LSTM:
    
    def __init__(self, n_concat, hidden_dim, device='cpu'):
        
        self.n_concat = n_concat
        
        # initialize bi-directional LSTM and mount onto the assigned device
        self.nn = LSTM(13 * n_concat, hidden_dim, device)
        self.to(device)
        
        
    def fit(self, dset_trn, dset_vld=None, n_epoch=8, b_size=32, lr=.001, save_model=False):
        
        # initialize data loaders
        kwargs = {'batch_size': b_size,
                  'shuffle': True,
                  'num_workers': 1,
                  'collate_fn': collate_fn}
        dldr_trn = data.DataLoader(dset_trn, **kwargs)
        
        # initialize loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # for model selection
        vld_mcc = -1
        
        for epoch in range(n_epoch):
            
            # set model to training mode
            self.nn.train()
            
            # model performance statistics
            cum_loss, cum_corr, count = 0, 0, 0
            
            # training loop
            with tqdm(total=len(dset_trn), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, bar_format='{l_bar}{r_bar}', file=sys.stdout) as pbar:
                for Xs, ys, _ in dldr_trn:
                    
                    # mount data to device
                    for i in range(len(Xs)):
                        Xs[i] = reshape_(Xs[i], self.n_concat, 200)
                        Xs[i] = torch.tensor(Xs[i], dtype=torch.float32, device=self.nn.device)
                        
                    ys = torch.tensor(ys, dtype=torch.long, device=self.nn.device)
                    
                    # forward and backward propagation
                    self.nn.zero_grad()
#                    scores = self.nn(Xs)
                    scores, scores_seg = self.nn(Xs)
                    
                    # loss
                    loss = loss_fn(scores, ys) * .1
                    
                    for i in range(len(scores_seg)):
                        ys_seg = torch.ones(scores_seg[i].shape[0], dtype=torch.long, device=self.nn.device) * ys[i]
                        loss += loss_fn(scores_seg[i], ys_seg)
                    
                    loss.backward()
                    op.step()
                    
                    # predictions
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
            with tqdm(total=len(dset), desc='Epoch ___ (EVL)', ascii=True, bar_format='{l_bar}{r_bar}', file=sys.stdout) as pbar:
                for Xs, _, _ in dldr:    
                    # mount data to device
                    for i in range(len(Xs)):
                        Xs[i] = reshape_(Xs[i], self.n_concat, 200)
                        Xs[i] = torch.tensor(Xs[i], dtype=torch.float32, device=self.nn.device)
                    
                    # forward propagation
#                    out = self.nn(Xs)
                    out, _ = self.nn(Xs)
                    
                    # append batch outputs to result
                    rsl.append(out.data.cpu().numpy())
                    
                    # progress bar
                    pbar.update(len(Xs))
        
        # concatenate all batch outputs
        rsl = np.concatenate(rsl)
        
        return rsl

    
    def prob(self, dset, b_size=32):
        
        # get network output
        rsl = self.eval(dset, b_size)
        
        # convert output to probability by softmax
        rsl = np.exp(rsl)[:,1] / np.sum(np.exp(rsl), axis=1)

        return rsl
                    
    
    def save_model(self, fp):
        torch.save(self.nn, fp)
        
    
    def load_model(self, fp):
        self.nn = torch.load(fp)
        

    def to(self, device):
        assert device in ['cpu', 0, 1, 2, 3], 'Invalid device.'
        self.nn.to(device)
        self.nn.device = device