#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:57:49 2020

@author: cxue2
"""
import os
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
        get_trn_test = kwargs.get('get_trn_test_func', fhs_sdf.get_trn_test)
        get_trn_test_kw = kwargs.get('get_trn_test_func_kw', {})
        get_samples = kwargs.get('get_samples', fhs_sdf.get_samples)
        get_samples_kw = kwargs.get('get_samples_kw', {})
        create_folds = kwargs.get('create_folds', fhs_sdf.create_folds)
        create_folds_kw = kwargs.get('create_folds_kw', {})
        get_label = kwargs.get('get_label', lambda r: int(r['is_demented_at_recording']))
        get_label_kw = kwargs.get('get_label_kw', {})
        get_audio = kwargs.get('get_audio', lambda r: r['mfcc_npy_files']\
            .strip('[]').replace('\'', '').split(', '))
        get_audio_kw = kwargs.get('get_audio_kw', {})
        # assertions
        assert mode in ['TRN', 'VLD', 'TST']
        # instance variables
        self.mode = mode
        self.df = None
        # read csv file
        df_raw = pd.read_csv(csv, dtype=object)
        df_test, df_other = get_trn_test(df_raw, **get_trn_test_kw)
        # list of all unique patients
        lst_p_all = get_samples(df_raw, df_other, holdout_test, **get_samples_kw)
        print('# of unique patients found in the csv file: {}'.format(len(lst_p_all)))

        fld = create_folds(lst_p_all, num_folds, seed, **create_folds_kw)
        # split dataset
        if not holdout_test:
            lst_p = fhs_sdf.get_fold(lst_p_all, fld, vld_idx, tst_idx, mode)
        else:
            lst_p = fhs_sdf.get_holdout_fold(lst_p_all, df_test, fld, vld_idx, mode)

        print('Dataset mode: \'{}\''.format(mode))
        print('NumPy random seed: {}'.format(seed))
        print('# of the selected patients: {}'.format(len(np.unique(lst_p))))
        print(lst_p)
        lst_p_a_l = [] # list to hold [<pid>, <audio>, <label>]
        set_p = set(lst_p)
        for _, row in df_raw.iterrows():
            # patient id
            pid = '{}-{}'.format(row.idtype, row.id)
            # continue if patient id doesn't match
            if pid not in set_p:
                continue
            # label
            lbl = get_label(row, **get_label_kw)
            # audio filename (convert to list; possibly more than 1 file)
            fns = get_audio(row, **get_audio_kw)

            transcript_fns = row['duration_csv_out_list']
            transcript_fns = transcript_fns.replace('\\', '/')
            transcript_fns = transcript_fns.strip('[]').replace('\'', '').split(', ')
            has_transcripts = transcript_fns != [""]
            if has_transcripts:
                assert len(transcript_fns) == len(fns), f"{fns}, {transcript_fns}"
            for idx, fn in enumerate(fns):
                transcript = transcript_fns[idx] if has_transcripts else ""
                # check if file exists
                if os.path.exists(fn):
                    if transcript != "":
                        assert os.path.exists(transcript), transcript
                    tmp = [pid, fn, lbl, transcript]
                    lst_p_a_l.append(tmp)
                else:
                    print('Warning: {}.npy not found.'.format(fn[:-4]))
        print('# of associated audio files: {}'.format(len(lst_p_a_l)))
        self.num_patients = len(set(lst_p))
        self.num_audio = len(lst_p_a_l)
        self.num_positive_audio = sum([n for _, _, n, _ in lst_p_a_l])
        self.num_negative_audio = sum([1 for _, _, n, _ in lst_p_a_l if int(n) == 0])
        self.patient_list = list(set(lst_p))
        self.patient_list.sort()
        # print('Loading and processing audio files...')
        lst = []
        for pat, afn, lbl, tfn in lst_p_a_l:
            lst.append((pat, afn, lbl, tfn))
        # save to dataframe
        self.df_dat = pd.DataFrame(lst, columns=['patient_id', 'audio_fn', 'label',
            'transcript_fn'])

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
