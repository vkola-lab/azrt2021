#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:35:54 2020

@author: cxue2
"""

import glob, os, errno
from scipy.io import wavfile
import numpy as np

dir_wav = '/data/datasets/dVoice'
dir_tmp = '/data/datasets/dVoice/tmp'
dir_raw = '{}/npy_raw'.format(dir_tmp)
dir_sln = '{}/npy_de_slience'.format(dir_tmp)

# create temporary folders
try:
    os.makedirs(dir_raw)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise
        
try:
    os.makedirs(dir_sln)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

# find all *.wav files
lst_fn = [fn for fn in glob.glob(dir_wav + '/**/*.wav', recursive=True)]
print('{} *.wav files found.'.format(len(lst_fn)))

# convert wav files to numpy and save
print('Reading *.wav and converting to *.npy...')

for fn in lst_fn:
    
    print('Processing {}... '.format(fn), end='')
    
    # read *.wav file
    fs, data = wavfile.read(fn)
    
    # warn if frequency != 8000
    if fs != 8000: print('Warning: the sampling rate of {} is not 8000Hz.'.format(data))
    
    # name of the *.npy file to save
    fn_save = '{}/{}.npy'.format(dir_raw, os.path.split(fn)[1][:-4])
    
    # delete if the file exists
    if os.path.exists(fn_save):
        os.remove(fn_save)
        
    # save as *.npy
    np.save(fn_save, data)
    
    print('Done.')
    
print('Done.')