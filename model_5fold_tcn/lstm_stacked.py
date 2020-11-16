#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 04:41:50 2020

@author: cxue2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:25 2019

@author: cxue2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np

class LSTM_Bi(nn.Module):

    def __init__(self, input_dim, hidden_dim, device):
        super(LSTM_Bi, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # output dim for linear embedding
#        self.emb_dim = 2 ** int(np.ceil(np.log(input_dim)))
#        self.emb_dim = max(64 , self.emb_dim)  # minimum dim is 64
#        self.emb_dim = min(256, self.emb_dim)  # maximum dim is 256
        self.emb_dim = 64
        
#        self.bn = nn.BatchNorm1d(num_features=self.input_dim)
        self.emb = nn.Linear(input_dim, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.dp1 = nn.Dropout(p=.5)
        self.dp2 = nn.Dropout(p=.5)
        self.dp3 = nn.Dropout(p=.5)
        
        
    def forward(self, Xs):

        # dimensions for input
        batch_size, seq_len, input_dim = Xs.shape
        
        # linear embedding
        Xs = self.emb(Xs.view(-1, input_dim)).view(batch_size, seq_len, -1)
        
        # feed the lstm by the input
        ini_hc_state = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                        torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        # lstm
        lstm_out, _ = self.lstm(Xs, ini_hc_state)
        
        # combine backward and forward results
        lstm_out = lstm_out.view(batch_size, seq_len, 1, self.hidden_dim)
        lstm_out_f = lstm_out[:,-1,0,:]
#        lstm_out_b = lstm_out[:,-1,1,:]
        lstm_out_valid = lstm_out_f #+ lstm_out_b
        
#        print(lstm_out_valid.shape)
        
        # lstm hidden state to output space
        out = self.dp1(lstm_out_valid)
        out = F.relu(self.fc1(out))
        out = self.dp2(lstm_out_valid)
        out = F.relu(self.fc2(out))
        out = self.dp3(lstm_out_valid)
        out = self.fc3(out)
        
        return out
    
    
    def set_param(self, param_dict):
        try:
            for pn, _ in self.named_parameters():
                exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            self.input_dim = param_dict['input_dim'] 
            self.hidden_dim = param_dict['hidden_dim']        
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')
            
    
    def get_param(self):
        param_dict = {}
 
        # pytorch tensors
        for pn, pv in self.named_parameters():
            param_dict[pn] = pv.data.cpu().numpy()
            
        # hyperparameters
        param_dict['input_dim'] = self.input_dim
        param_dict['hidden_dim'] = self.hidden_dim

        return param_dict
        