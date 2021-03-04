#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:57:49 2020

@author: cxue2
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import fhs_split_dataframe as fhs_sdf

class AudioDataset(Dataset):
    """dVoice dataset."""

    def __init__(self, csv, mode, **kwargs):
        """
        init function;
        """
        num_folds = kwargs.get('num_folds', 5)
        vld_idx = kwargs.get('vld_idx')
        tst_idx = kwargs.get('tst_idx')
        seed = kwargs.get('seed', 3227)
        holdout_test = kwargs.get('holdout_test', False)

        get_all_trn_test = kwargs.get('get_all_trn_test_func', fhs_sdf.get_all_trn_test)
        get_all_trn_test_kw = kwargs.get('get_all_trn_test_func_kw', {})

        get_sample_ids = kwargs.get('get_sample_ids', fhs_sdf.get_sample_ids)
        get_sample_ids_kw = kwargs.get('get_sample_ids_kw', {})

        create_folds = kwargs.get('create_folds', fhs_sdf.create_folds)
        create_folds_kw = kwargs.get('create_folds_kw', {})

        get_pid = kwargs.get('get_pid', lambda r: f'{r.idtype}-{r.id}')
        get_pid_kw = kwargs.get('get_pid_kw', {})

        get_label = kwargs.get('get_label', lambda r: int(r['is_demented_at_recording']))
        get_label_kw = kwargs.get('get_label_kw', {})

        get_files = kwargs.get('get_files', lambda r: r['mfcc_npy_files']\
            .strip('[]').replace('\'', '').split(', '))
        get_files_kw = kwargs.get('get_files_kw', {})

        get_row_data = kwargs.get('get_row_data', fhs_sdf.get_row_data)
        get_row_data_kw = kwargs.get('get_row_data_kw', {})

        data_headers = kwargs.get('data_headers', ['patient_id', 'audio_fn', 'label',
            'transcript_fn'])
        # assertions
        assert mode in ['TRN', 'VLD', 'TST'], mode
        # instance variables
        self.mode = mode
        self.df = None
        # read csv file
        df_raw = pd.read_csv(csv, dtype=object)
        all_ids, test_ids, other_ids = get_all_trn_test(df_raw, **get_all_trn_test_kw)
        # list of all unique patients
        sample_ids = get_sample_ids(all_ids, other_ids, holdout_test, **get_sample_ids_kw)
        print('# of unique patients found in the csv file: {}'.format(len(sample_ids)))

        folds = create_folds(sample_ids, num_folds, seed, **create_folds_kw)
        # split dataset
        if not holdout_test:
            current_mode_ids = fhs_sdf.get_fold(sample_ids, folds, vld_idx, tst_idx, mode)
        else:
            current_mode_ids = fhs_sdf.get_holdout_fold(sample_ids, test_ids, folds, vld_idx, mode)

        print('Dataset mode: \'{}\''.format(mode))
        print('NumPy random seed: {}'.format(seed))
        print('# of the selected patients: {}'.format(len(np.unique(current_mode_ids))))
        print(current_mode_ids)
        data_list = [] # get data from each row;
        set_p = set(current_mode_ids)
        for _, row in df_raw.iterrows():
            # patient id
            pid = get_pid(row, **get_pid_kw)
            if pid not in set_p:
                continue
            lbl = get_label(row, **get_label_kw)
            fns = get_files(row, **get_files_kw)
            data_list.extend(get_row_data(row, pid, lbl, fns, **get_row_data_kw))
        print(f'# of associated audio files: {len(data_list)}')
        self.num_patients = len(set(current_mode_ids))
        self.num_audio = len(data_list)
        self.num_positive_audio = sum([n for _, _, n, _ in data_list])
        self.num_negative_audio = sum([1 for _, _, n, _ in data_list if int(n) == 0])
        self.patient_list = list(set(current_mode_ids))
        self.patient_list.sort()

        # save to dataframe
        self.df_dat = pd.DataFrame(data_list, columns=data_headers)

    def __len__(self):
        return len(self.df_dat)

    def __getitem__(self, idx):
        fea = np.load(self.df_dat.loc[idx, 'audio_fn'])
        return fea, self.df_dat.loc[idx,'label'], self.df_dat.loc[idx,'patient_id']

    @property
    def labels(self):
        """
        convert label column to np array;
        """
        return self.df_dat.label.to_numpy()

    @property
    def df_sampling_weights(self):
        """
        convert label to numpy() and add 1 to each;
        """
        return self.df_dat.label.to_numpy() + 1

    @property
    def audio_fns(self):
        """
        convert audio filename columns to np array;
        """
        return self.df_dat['audio_fn'].to_numpy()

    @property
    def transcript_fns(self):
        """
        convert transcript filename columns to np array;
        """
        return self.df_dat['transcript_fn'].to_numpy()

def collate_fn(batch):
    """
    collect audio path, label, patient ID
    """
    aud = [itm[0] for itm in batch]
    lbl = np.stack([itm[1] for itm in batch])
    pid = np.stack([itm[2] for itm in batch])
    return aud, lbl, pid
