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
import torch
import cupy as cp
import sys
import random
    
device = 2

dir_mfcc = None
# csv_info = '/encryptedfs/scripts/dvoice_lstm/dvoice_labels/rekey_revalue/2020102121/rekey_revalue_(485)_[813]_2020102121_dvoice_dr_mci_or_null_excluded_add_mfccs.csv'

if len(sys.argv) == 1:
    print("0=norm vs. demented;\n1=nondemented vs. demented;\n2=norm vs. mci;\n3=mci vs. demented;")
    sys.exit()
task_id = int(sys.argv[1])
get_label = None
if task_id == 0:
    ext = "norm_vs_demented"
    # csv_info = '/encryptedfs/scripts/dvoice_lstm/dvoice_labels/rekey_revalue/2020102121/rekey_revalue_(485)_[813]_2020102121_dvoice_dr_mci_or_null_excluded_add_mfccs.csv'
    csv_info = '/encryptedfs/scripts/dvoice_lstm/dvoice_labels/iterative_reduce/20210111_12_4_42_0311/iterative_reduce_(485)_[813]_20210111_12_4_42_0311_attach_transcripts.csv'
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

weights = [1, 1]
if weights != []:
    ext += "_with_sampling_weights_1_2_two_thirds_sample_size"
    # ext += "_with_loss_weights_1_2_v3"

n_epoch = 128
do_random = True
get_dir_rsl = lambda e, n, s: f'results/holdout_test_{e}/{n}_epochs/{s}'

if not do_random:
    seed_list = [21269, 19952]
else:
    seed_list = [21269, 19952]
    for i in range(10):
        seed = random.randint(0, 100000)
        dir_rsl = get_dir_rsl(ext, n_epoch, seed)
        while os.path.isdir(dir_rsl):
            seed = random.randint(0, 100000)
            dir_rsl = get_dir_rsl(ext, n_epoch, seed)
        seed_list.append(seed)

seed_to_dir = {s: get_dir_rsl(ext, n_epoch, s) for s in seed_list}

for seed, dir_rsl in seed_to_dir.items():
    # dir_rsl = f'results/holdout_test_{ext}/{n_epoch}_epochs/{seed}'
    assert not os.path.isdir(dir_rsl), dir_rsl
        
    # create folder for saving results
    try:
        os.makedirs(dir_rsl)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
        
    for i in range(5):
        kwargs = {'comb': (i,), 'seed': seed, 'holdout_test': True}
        dset_trn = AudioDataset(dir_mfcc, csv_info, 'TRN', **kwargs)
        dset_vld = AudioDataset(dir_mfcc, csv_info, 'VLD', **kwargs)
        dset_tst = AudioDataset(dir_mfcc, csv_info, 'TST', **kwargs)
         
        # initialize model
        model = Model_LSTM(n_concat=10, hidden_dim=64, device=device)
            
        # train model
        model.fit(dset_trn, dset_vld=dset_vld, n_epoch=n_epoch, b_size=4, lr=1e-4, save_model=False, weights=weights)
        if not os.path.isdir(f"pt_files/{ext}"):
            os.makedirs(f"pt_files/{ext}")
        model.save_model(f"./pt_files/{ext}/{ext}_holdout_model_{i}_{seed}_{n_epoch}_epochs.pt")    
        # evaluate model on validation dataset
        rsl = model.prob(dset_tst, b_size=64)
        # break
        # save result to dataframe
        df_dat = dset_tst.df_dat
        df_dat['score'] = rsl
            
        # save dataframe to csv
        df_dat.to_csv('{}/audio_{}_{}.csv'.format(dir_rsl, seed, i), index=False)
        txt_fp = os.path.join(dir_rsl, f"comb_[{i}].txt")
        lines = []
        for dset_ext, dataset in [('TRN',dset_trn), ('VLD', dset_vld), ('TST', dset_tst)]:
            line = f"{dset_ext}: num_patients: {dataset.num_patients}, num_audio: {dataset.num_audio} [normal={dataset.num_normal_audio}, demented={dataset.num_demented_audio}]\n"
            lines.append(line)
        with open(txt_fp, 'w') as outfile:
            outfile.write(f"holdout_train; seed={seed}; i={i}; loss_weights={str(weights)};\n")
            for line in lines:
                outfile.write(line)
            outfile.write(f"\nTRN IDs: {dset_trn.patient_list}\n\n")
            outfile.write(f"VLD IDs: {dset_vld.patient_list}\n\n")
            outfile.write(f"TST IDs: {dset_tst.patient_list}\n\n")
        
    
        # break
