#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 03:00:04 2020

@author: cxue2
"""

import numpy as np
import cupy as cp

class MFCC_Extractor():
    
    def __init__(self, samplerate=16000, winlen=.025, winstep=.01, numcep=13,
                 nfilt=26, nfft='POW2', lowfreq=0, highfreq=None, preemph=.97,
                 ceplifter=22, appendEnergy=True, winfunc=None, device='cpu'):
        
        assert nfft in ['WINLEN', 'POW2']
        
        # parameters
        self.samplerate = samplerate
        self.winlen     = int(winlen  * samplerate)
        self.winstep    = int(winstep * samplerate)
        self.numcep     = numcep
        self.nfilt      = nfilt
        self.preemph    = preemph
        self.ceplifter  = ceplifter
        
        self.nfft = int(2 ** np.ceil(np.log2(self.winlen))) if nfft == 'POW2' else self.winlen
        
        # construct Mel filter bank
        self.bnk = self._mel_filter_bank()
        self.bnk = self.bnk.astype(np.float32)
        
        # DCT matrix (type 2)
        self.dct_mat = self._dct_mat_type_2()
        self.dct_scl = np.zeros((nfilt,), dtype=np.float32) # for orthogonal transformation
        self.dct_scl[0]  = np.sqrt(1 / 4 / nfilt)
        self.dct_scl[1:] = np.sqrt(1 / 2 / nfilt)
        
        # cepstrum lifter
        self.lft = 1 + (ceplifter / 2) * np.sin(np.pi * np.arange(numcep) / ceplifter)
        self.lft = self.lft.astype(np.float32)
        
        # set device
        self.device = device
        
        if device == 'cpu':
            self.backend = np
            
        else:
            cp.cuda.runtime.setDevice(device)
            self.backend = cp
            
            # transfer operators to GPU
            self.bnk     = cp.asarray(self.bnk)
            self.dct_mat = cp.asarray(self.dct_mat)
            self.dct_scl = cp.asarray(self.dct_scl)
            self.lft     = cp.asarray(self.lft)
            
    
    def __call__(self, arr):
        '''
        Extract MFCC features.

        Parameters
        ----------
        arr : numpy.ndarry of shape (batch_size, audio_length)
            Audio sequences.

        Returns
        -------
        numpy.ndarry of shape (batch_size, # of chunks, # of ceps)
            MFCC features.
        '''
        
        # mount to device
        tmp = np.copy(arr) if self.device == 'cpu' else cp.asarray(arr)
        
        # flatten array except the last dimension
        shp = tmp.shape
        tmp = tmp.reshape((-1, shp[-1]))
        
        # step 0: pre-emphasis
        tmp[:,1:] = tmp[:,1:] - self.preemph * tmp[:,:-1]
        
        # step 1: split audio into chunks
        # note: all batches are merged
        tmp, seq_len = self._strided_split(tmp)
        
        # step 2.1: apply Hamming window
        # tmp = tmp * self.backend.hamming(winlen)
        
        # step 2.2: apply FFT
        tmp = self.backend.fft.rfft(tmp, n=self.nfft, norm='ortho')
        
        # step 2.3: to power spectrum
        tmp = self.backend.real(tmp * tmp.conj()) / self.nfft
        
        # tatal energy
        eng = self.backend.sum(tmp, 1) * self.nfft
        eng = self.backend.where(eng == 0, np.finfo(np.float32).eps, eng)  # numerical stability for log
        
        # step 3: apply Mel filter bank
        tmp = tmp @ self.bnk.T        
        tmp = self.backend.where(tmp == 0, np.finfo(np.float32).eps, tmp)  # numerical stability for log
        tmp = self.backend.log(tmp)
        
        # step 4.1: DCT
        tmp = tmp @ self.dct_mat.T
        tmp *= self.dct_scl
        
        # step 4.2: no. of cepstrum coefficients to return
        tmp = tmp[:,:self.numcep]
        
        # apply a lifter to final cepstral coefficients
        tmp *= self.lft
        
        # replace the 0th cep coefficient by log total energy
        tmp[:,0] = self.backend.log(eng)
        
        # reshape
        tmp = tmp.reshape(shp[:-1] + (seq_len, -1))
        
        # clean up GPU memory
#        if self.device != 'cpu':
#            cp.get_default_memory_pool().free_all_blocks()
        
        return tmp

    
    def _strided_split(self, arr):
        
        # sequence length
        seq_len = (arr.shape[-1] - self.winlen) // self.winstep + 1
        
        # data size in bytes
        d_size = arr.strides[-1]
        
        # result placeholder
        rsl = self.backend.empty((seq_len * len(arr), self.winlen), dtype=arr.dtype)
        
        for i, seq in enumerate(arr):
            rsl[i*seq_len:(i+1)*seq_len] = self.backend.lib.stride_tricks.as_strided(seq, shape=(seq_len, self.winlen), strides=(self.winstep*d_size, d_size))
        
        return rsl, seq_len
    
    
    def _mel_filter_bank(self):
        
        # placeholder for filter bank
        bnk = np.zeros((self.nfilt, int(np.floor(self.nfft / 2 + 1))), dtype=np.float32)
        
        frq_mel = (0, 2595 * np.log10(1 + (self.samplerate / 2) / 700))  # (low, high)
        mel_pts = np.linspace(frq_mel[0], frq_mel[1], self.nfilt + 2)    # equally spaced in Mel scale
        hz__pts = 700 * (10 ** (mel_pts / 2595) - 1)                     # convert Mel to Hz
        idc     = np.floor((self.nfft + 1) * hz__pts / self.samplerate).astype(np.int)
        
        for m in range(1, self.nfilt + 1):
            f_m_l = idc[m-1]  # left
            f_m_c = idc[m]    # center
            f_m_r = idc[m+1]  # right
        
            for k in range(f_m_l, f_m_c):
                bnk[m-1,k] = (k - idc[m-1]) / (idc[m] - idc[m-1])
                
            for k in range(f_m_c, f_m_r):
                bnk[m-1,k] = (idc[m+1] - k) / (idc[m+1] - idc[m])
                
        return bnk
    
    
    def _dct_mat_type_2(self):
        
        # placeholder for dct matrix
        mat = np.zeros((self.nfilt, self.nfilt), dtype=np.float32)
        
        for k in range(self.nfilt):
            mat[k,:] = np.pi * k * (2 * np.arange(self.nfilt) + 1) / (2 * self.nfilt)
            mat[k,:] = 2 * np.cos(mat[k,:])
            
        return mat
    
    
if __name__ == '__main__':
    
    from data import AudioDataset
    
    dir_audio = '/data_2/dVoice/tmp/npy_raw'
    csv = './data/rekey_revalue_(89)_[135]_2020072101_lstm_labels_threshold_[180]_days_paths.csv'
    ds = AudioDataset(dir_audio, csv, 'VLD', preload=False)
    
    psf_mfcc = lambda aud: psf.mfcc(aud, samplerate=8000, winlen=.025, winstep=.01, numcep=13, nfilt=26, nfft=256, preemph=0.97, ceplifter=22)
    
    # new extractor
    ext = MFCC_Extractor(device=0)
#    print(ext(ds[0][0])[:100])
    
    import python_speech_features as psf
#    
#    tmp = psf.mfcc(ds[0][0], samplerate=8000, winlen=.025, winstep=.01, numcep=13,
#                   nfilt=26, nfft=256, preemph=0.97, ceplifter=22)[:100]
#    print(tmp)
#    
    
    import timeit
    import matplotlib.pyplot as plt
    
    l = ds[0][0].shape[-1]

    N = 32
    arr = np.zeros((N, l), dtype=np.float32)
    for i in range(N):
        arr[i,:] = ds[i][0] 
    
    n_iter = 100
    starttime = timeit.default_timer()
    for _ in range(n_iter):
        rsl = ext(arr)
#        rsl = psf_mfcc(ds[0][0])
    print('The avg. time:', (timeit.default_timer() - starttime) / N / n_iter)
    
    cp.get_default_memory_pool().free_all_blocks()
    
    print(rsl.shape)
    
    # 0.00036871268559480086
#    print(rsl)
    
#    plt.matshow(ds[0][0][:100,:].T, cmap='hot')
    
#    for i in range(N):
#        print(cp.asnumpy(rsl[i][:]) / psf_mfcc(ds[i][0])[:-1])