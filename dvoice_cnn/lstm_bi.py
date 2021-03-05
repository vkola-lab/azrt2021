#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:25 2019

@author: cxue2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from data import reshape_

class LSTM(nn.Module):
    """
    lstm model
    """

    def __init__(self, dim_i, dim_h, device):
        """
        init;
        """
        super(LSTM, self).__init__()
        self.device = device     # 'cpu' or 'cuda:x'

        # embbedding layer
        dim_e = 64
        self.emb = nn.Linear(dim_i, dim_e)

        # 2-level stacked lstm
        self.lstm_lv1 = _Module_LSTM(dim_e, dim_h, device)
        self.lstm_lv2 = _Module_LSTM(dim_h, dim_h, device)

        # fully connected layers
        self.fc1 = nn.Linear(dim_h, dim_h)
        self.fc2 = nn.Linear(dim_h, 2)
        self.dp1 = nn.Dropout(p=.5)
        self.dp2 = nn.Dropout(p=.5)

        # fully connected layers for segments
        self.fc1_ = nn.Linear(dim_h, dim_h)
        self.fc2_ = nn.Linear(dim_h, 2)
        self.dp1_ = nn.Dropout(p=.5)
        self.dp2_ = nn.Dropout(p=.5)

    def forward(self, Xs):
        '''
        Parameters
        ----------
        Xs : Qsys
            Right operand. It is supposed to be another Qsys or Qbit.

        Returns
        -------
        Qsys
            Tensor product.
        '''
        # output placeholder
        out, out_ = [], []
        for tmp in Xs:
            # get sequence dimensions
            l2, l1, l0 = tmp.shape
            # linear embedding
            tmp = tmp.reshape(l2 * l1, l0)
            tmp = self.emb(tmp)
            tmp = tmp.reshape(l2, l1, tmp.shape[-1])
            # level 1 lstm
            tmp = self.lstm_lv1(tmp)
            out_.append(self.fc2_(self.dp2_(F.relu(self.fc1_(self.dp1_(tmp))))))
            # level 2 lstm
            tmp = tmp.unsqueeze(0)
            tmp = self.lstm_lv2(tmp)
            # fully connected layers
            tmp = self.fc1(self.dp1(tmp))
            tmp = F.relu(tmp)
            tmp = self.fc2(self.dp2(tmp))
            # append to output
            out.append(tmp.squeeze())
        return torch.stack(out), out_

    def get_scores_loss(self, Xs, ys, loss_fn):
        """
        get scores and loss
        """
        scores, scores_seg = self(Xs)
        loss = loss_fn(scores, ys) * .1
        for i in range(len(scores_seg)):
            ys_seg = torch.ones(scores_seg[i].shape[0],
                dtype=torch.long, device=self.nn.device) * ys[i]
            loss += loss_fn(scores_seg[i], ys_seg)
        return scores, loss

    def reformat(self, Xs, n_concat):
        """
        reformat Xs
        """
        for i in range(len(Xs)):
            Xs[i] = reshape_(Xs[i], n_concat, 200)
            Xs[i] = torch.tensor(Xs[i], dtype=torch.float32, device=self.nn.device)

class _Module_LSTM(nn.Module):
    """
    individual lstm class;
    """
    def __init__(self, dim_i, dim_h, device):
        """
        init
        """
        super(_Module_LSTM, self).__init__()
        self.device = device
        self.dim_i = dim_i    # input dimension
        self.dim_h = dim_h    # hidden state dimension
        # lstm
        self.lstm = nn.LSTM(dim_i, dim_h, batch_first=True, bidirectional=True)

    def forward(self, Xs):
        """
        forward;
        """
        # dimensions for input
        batch_size, seq_len, input_dim = Xs.shape
        # feed the lstm by the input
        ini_hc_state = (torch.zeros(2, batch_size, self.dim_h).to(self.device),
                        torch.zeros(2, batch_size, self.dim_h).to(self.device))
        # lstm
        lstm_out, _ = self.lstm(Xs, ini_hc_state)
        # combine backward and forward results
        lstm_out = lstm_out.view(batch_size, seq_len, 2, self.dim_h)
        lstm_out_f = lstm_out[:,-1,0,:]
        lstm_out_b = lstm_out[:,-1,1,:]
        out = lstm_out_f + lstm_out_b
        return out
