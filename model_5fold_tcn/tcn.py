# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:52:17 2020

@author: Iluva
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np


class TCN(nn.Module):
    
    
    def __init__(self, dim_i, device):
        
        super(TCN, self).__init__()
        self.device = device     # 'cpu' or 'cuda:x'
        
        # TCN
        self.tcn = nn.Sequential(
            # nn.BatchNorm1d(13),
            nn.Conv1d(in_channels=13, out_channels=32, kernel_size=1, stride=1, padding=0),
            
            # nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            # nn.Dropout(),
            
            # nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            # nn.Dropout(),
            
            # nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            
            # nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            
            # nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            
            # nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            
            # nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            
            # nn.BatchNorm1d(512),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        
        # linear layer
        self.mlp = nn.Sequential(
            nn.Linear(512, 2, bias=False)
        )
        
        self.tcn.to(device)
        self.mlp.to(device)
        
        
    def forward(self, Xs):
        
        out = []
        
        for X in Xs:
            tmp = self.tcn(X)
            
            # global average pooling
            tmp = torch.mean(tmp, dim=2)
            
            # linear layer
            tmp = self.mlp(tmp)
            
            tmp = tmp.squeeze()
            out.append(tmp)
        
        out = torch.stack(out)
            
        return out
    
    
if __name__ == '__main__':
    
    i = [torch.rand(1, 13, 160000).to(0),
         torch.rand(1, 13, 40000).to(0)]
    m = TCN(13, device=0)
    o = m(i)
    
    print(o)
    print(o.shape)