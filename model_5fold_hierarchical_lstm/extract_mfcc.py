# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:03:38 2020

@author: Iluva
"""

dir_from = '/data_2/dVoice/tmp/npy_raw'
dir_to = '/data_2/dVoice/tmp/npy_mfcc'

import python_speech_features_cuda as psf
import os
import cupy as cp
import numpy as np
import os, errno
from tqdm import tqdm

psf.env.backend = cp
psf.env.dtype = np.float32

lst_fn = os.listdir(dir_from)
    
 # create folder for dir_to
try:
    os.makedirs(dir_to)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

for fn in tqdm(lst_fn):
    
    # load audio files
    aud = np.load('{}/{}.npy'.format(dir_from, fn[:-4]))
    aud = aud.astype(np.float32)
    
    # to gpu
    aud = cp.asarray(aud)
    
    # extract mfccs
    fea = psf.mfcc(aud, samplerate=8000, nfft=512)
    psf.buf.reset()
    
    # to cpu
    fea = cp.asnumpy(fea)
    
    # save
    np.save('{}/{}.npy'.format(dir_to, fn[:-4]), fea)