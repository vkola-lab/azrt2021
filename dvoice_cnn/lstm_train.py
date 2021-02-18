# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:19:01 2018

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""
import os, errno
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # 

from lstm_model import Model_LSTM
from data import AudioDataset
from misc import calc_performance_metrics, show_performance_metrics
from time import time
from datetime import datetime
import cupy as cp
import sys
    
device = 0

dir_mfcc = None
# csv_info = '/encryptedfs/scripts/dvoice_lstm/dvoice_labels/rekey_revalue/2020102121/rekey_revalue_(485)_[813]_2020102121_dvoice_dr_mci_or_null_excluded_add_mfccs.csv'

if len(sys.argv) == 1:
    print("0=norm vs. demented;\n1=nondemented vs. demented;\n2=norm vs. mci;\n3=mci vs. demented;")
    sys.exit()
task_id = int(sys.argv[1])
get_label = None
if task_id == 0:
    ext = "norm_vs_demented"
    csv_info = '/encryptedfs/scripts/dvoice_lstm/dvoice_labels/rekey_revalue/2020102121/rekey_revalue_(485)_[813]_2020102121_dvoice_dr_mci_or_null_excluded_add_mfccs.csv'
elif task_id == 1:
    ext = "nondemented_vs_demented"
    csv_info = "/encryptedfs/scripts/dvoice_lstm/dvoice_labels/rekey_revalue/2020110429/rekey_revalue_(656)_[1264]_2020110429_dvoice_dr_nondemented_demented_add_mfccs.csv"
elif task_id == 2:
    ext = "norm_vs_mci"
    csv_info = "/encryptedfs/scripts/dvoice_lstm/dvoice_labels/filter_data/2020110416/filter_data_(507)_[934]_2020110416_norm_and_mci_with_mfccs_passed_filter.csv"
    get_label = lambda r: 1 if float(r['most_severe_closest']) == 0.5 else 0
elif task_id == 3:
    ext = "mci_vs_demented"
    csv_info = "/encryptedfs/scripts/dvoice_lstm/dvoice_labels/filter_data/2020110459/filter_data_(476)_[781]_2020110459_mci_and_demented_with_mfccs_passed_filter.csv"
else:
    ext = None
    csv_info = None
    sys.exit()

n_epoch = 4
# seed = 65779
seed = 1000
# dir_rsl = f'results/{seed}_{n_epoch}epochs'

dir_rsl = f'results/{ext}/{n_epoch}_epochs/{seed}'
assert not os.path.isdir(dir_rsl), dir_rsl

# create folder for saving results
try:
    os.makedirs(dir_rsl)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise
    
for i in range(5):
    for j in range(5):
    
        # validation and testing dataset cannot be the same
        if i == j: continue
        
        dset_trn = AudioDataset(dir_mfcc, csv_info, 'TRN', comb=(i, j), seed=seed)
        dset_vld = AudioDataset(dir_mfcc, csv_info, 'VLD', comb=(i, j), seed=seed)
        dset_tst = AudioDataset(dir_mfcc, csv_info, 'TST', comb=(i, j), seed=seed)
        
        # initialize model
        model = Model_LSTM(n_concat=10, hidden_dim=64, device=device)
        
        # train model
        model.fit(dset_trn, dset_vld=dset_vld, n_epoch=n_epoch, b_size=4, lr=1e-4, save_model=False)
        
        # evaluate model on validation dataset
        rsl = model.prob(dset_tst, b_size=64)
        # break
        # save result to dataframe
        df_dat = dset_tst.df_dat
        df_dat['score'] = rsl
        
        # save dataframe to csv
        df_dat.to_csv('{}/audio_{}_{}_{}.csv'.format(dir_rsl, seed, i, j), index=False)
    # break
