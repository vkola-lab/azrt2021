#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:57:49 2020

@author: cxue2
"""

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tqdm
import os
import sys
import random

class AudioDataset(Dataset):
    """dVoice dataset."""

    def __init__(self, dir_mfcc, csv, mode, comb=(0, 1), seed=3227):
        
        # assertions
        assert mode in ['TRN', 'VLD', 'TST']
        assert len(comb) == 2
        
        # instance variables
        self.mode = mode
        self.df = None
        
        # read csv file
        df_raw = pd.read_csv(csv, dtype=object)
        
        # list of all unique patients
        tmp_0 = df_raw.idtype.values.ravel('K')
        tmp_1 = df_raw.id.values.ravel('K')
        tmp_2 = ['{}-{}'.format(tmp_0[i], tmp_1[i]) for i in range(len(tmp_0))]
        lst_p_all = np.unique(tmp_2)  # remove duplication
        print('# of unique patients found in the csv file: {}'.format(len(lst_p_all)))
              
        # split patients into 5 folds.
        random.seed(seed)
        lst_idx = np.array(range(len(lst_p_all)))
        random.shuffle(lst_idx)
        fld = [lst_idx[np.arange(len(lst_p_all)) % 5 == i] for i in range(5)]
        
        # split dataset
        if mode == 'VLD':
            idx = fld[comb[0]]
        elif mode == 'TST':
            idx = fld[comb[1]]
        else:
            tmp = [0, 1, 2, 3, 4]
            tmp.remove(comb[0])
            tmp.remove(comb[1])
            idx = np.concatenate([fld[tmp[i]] for i in range(3)])
        lst_p = lst_p_all[idx]
        print('Dataset mode: \'{}\''.format(mode))
        print('NumPy random seed: {}'.format(seed))
        print('# of the selected patients: {}'.format(len(lst_p)))     
        print(lst_p)
        
        lst_p_a_l = []  # list to hold [<pid>, <audio>, <label>]
        set_p = set(lst_p)
        for _, row in df_raw.iterrows():
            # patient id
            pid = '{}-{}'.format(row.idtype, row.id)
            
            # continue if patient id doesn't match
            if pid not in set_p:
                continue
            
            # label
            lbl = int(row['is_demented_at_recording'])
            
            # audio filename (convert to list; possibly more than 1 file)
            fns = row['mfcc_npy_files']
            fns = fns.strip('[]').replace('\'', '').split(', ')
            # fns = [os.path.basename(fn) for fn in fns]
            
            for fn in fns:
            
                # check if file exists
                if os.path.exists(fn):
                    tmp = [pid, fn, lbl, self.mode]
                    lst_p_a_l.append(tmp)
                else:
                    print('Warning: {}.npy not found.'.format(fn[:-4]))
                    
        print('# of associated audio files: {}'.format(len(lst_p_a_l)))
        
        # print('Loading and processing audio files...')
        lst = []
        for pat, afn, lbl, mode in lst_p_a_l:
            lst.append((pat, afn, lbl, mode))
            
        # save to dataframe
        self.df_dat = pd.DataFrame(lst, columns=['patient_id', 'audio_fn', 'label', 'mode'])
      

    def __len__(self):
        
        return len(self.df_dat)
    

    def __getitem__(self, idx):
        fea = np.load(self.df_dat.loc[idx, 'audio_fn'])
        return fea, self.df_dat.loc[idx,'label'], self.df_dat.loc[idx,'patient_id']
    
    @property
    def labels(self):
        
        return self.df_dat.label.to_numpy()
    

def collate_fn(batch):

    aud = [itm[0] for itm in batch]
    lbl = np.stack([itm[1] for itm in batch])
    pid = np.stack([itm[2] for itm in batch])
    
    return aud, lbl, pid

    
if __name__ == '__main__':
    
    dir_mfcc = '/data_2/dVoice/tmp/npy_mfcc'
    csv = '../data/filter_data_(88)_[134]_2020092814_anon_data_passed_filter.csv'
    _ = AudioDataset(dir_mfcc, csv, 'VLD', comb=(0, 1))
    
    print(_.df_dat)
    print(_[0][0].shape)
    
