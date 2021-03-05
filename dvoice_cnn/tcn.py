# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:52:17 2020

@author: Iluva
"""
import torch
import torch.nn as nn

class TCN(nn.Module):
    """
    TCN class;
    """
    def __init__(self, device):
        """
        init method;
        """
        super(TCN, self).__init__()
        self.device = device     # 'cpu' or 'cuda:x'
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
        """
        pass fwd;
        """
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

    def forward_wo_gpool(self, Xs):
        """
        fwd without gpool;
        """
        out = []
        for X in Xs:
            tmp = self.tcn(X)
            out.append(tmp)
        return out

    def get_scores_loss(self, Xs, ys, loss_fn):
        """
        get scores and loss;
        """
        scores = self(Xs)
        loss = loss_fn(scores, ys)
        return scores, loss

    def reformat(self, Xs, _):
        """
        reformat Xs array accordingly;
        """
        for idx, _ in enumerate(Xs):
            Xs[idx] = torch.tensor(Xs[idx], dtype=torch.float32,
                device=self.device)
            Xs[idx] = Xs[idx].permute(1, 0)
            Xs[idx] = Xs[idx].view(1, Xs[idx].shape[0], Xs[idx].shape[1])
