#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:17:08 2020

@author: cxue2
"""
import sys
import pandas as pd
import numpy as np
import glob, os
from misc import calc_performance_metrics, show_performance_metrics
from misc import get_roc_info, get_pr_info
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp
from collections import defaultdict

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'

def plot_roc(curve, ax, xs, ys_mean, ys_upper, ys_lower, auc_mean, auc_std):
    
    assert curve in ['roc', 'pr']
    if curve == 'roc':
        ys_mean = ys_mean[::-1]
        ys_upper = ys_upper[::-1]
        ys_lower = ys_lower[::-1]
        xlabel, ylabel = 'Specificity', 'Sensitivity'
    else:
        xlabel, ylabel = 'Recall', 'Precision'

    p_mean, = ax.plot(
        xs, ys_mean, color='C0',
        linestyle='-',
        lw=1.5, alpha=1)

    p_fill = ax.fill_between(
        xs, ys_lower, ys_upper,
        alpha=.4,
        facecolor='none',
        edgecolor='C0',
        hatch='//////')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.xaxis.set_label_coords(0.5, -0.01)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.yaxis.set_label_coords(-0.01, 0.5)
    ax.set_title('', fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(ax.get_xticks(), weight='bold')

    ax.set_aspect('equal', 'box')
    ax.set_facecolor('w')
    plt.setp(ax.spines.values(), color='w')
    ax.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    ax.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    
    msg = 'AUC: {:.3f}$\pm${:.3f}'.format(auc_mean, auc_std)
    ax.legend([p_mean], [msg],
              facecolor='w', prop={"weight":'bold', "size":17},
              bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
              loc='lower left')

    return p_mean, p_fill
    

if __name__ == '__main__':
    
#    dir_rsl = './results/exp_2020_07_27_03_28'
#    dir_rsl = './results/exp_2020_07_30_03_05'
    # dir_rsl = './results/exp_2020_07_30_20_15'
#    dir_rsl = './results/exp_2020_07_30_23_28'
#    dir_rsl = './results/exp_2020_07_31_00_00'
#    dir_rsl = './results/exp_2020_07_31_00_57'
#    dir_rsl = './results/exp_2020_07_31_04_16'
    if len(sys.argv) == 3:
        dir_rsl, num_csvs = sys.argv[1:]
    else:
        dir_rsl = sys.argv[1]
        num_csvs = None
    mode = 'audio_avg'
    
    # list of all csv files
    if num_csvs is None:
        lst_csv = glob.glob(dir_rsl + '/*.csv', recursive=False)
        dirs_read = [dir_rsl]
    else:
        lst_csv = []
        dirs_read = []
        directories = [os.path.join(dir_rsl, d) for d in os.listdir(dir_rsl)]
        directories = [d for d in directories if os.path.isdir(d)]
        for directory in directories:
            current_lst = glob.glob(directory + '/*.csv', recursive=False)
            if len(current_lst) == int(num_csvs):
                lst_csv.extend(current_lst)
                dirs_read.append(directory)
    lst_lbl, lst_scr = [], []
    
    # all metrics
    mtr_all = defaultdict(list)
    assert lst_csv != [], dirs_read
    print(f"{len(lst_csv)} csvs found;")
    print("\n".join(dirs_read)) 
    for fn in lst_csv:
        
        fn_base = os.path.basename(fn)
        
        if not fn_base.startswith('audio'):
            continue
        
        # read from csv
        df = pd.read_csv(fn)
        
        # get scores and labels
        if mode == 'chunk':
            lbl = df.label.to_numpy()
            scr = df.score.to_numpy()
        elif mode == 'audio_avg':
            tmp = df.groupby('audio_fn').mean().to_numpy()
            lbl = tmp[:,0].astype(np.int)
            scr = tmp[:,-1]
        
        # metrics        
        mtr = calc_performance_metrics(scr, lbl)
#        print(fn_base)
#        show_performance_metrics(mtr)
        
        for k in mtr:
            if k == 'mat': continue
            mtr_all[k].append(mtr[k])
        
#        print()
        
        lst_lbl.append(lbl)
        lst_scr.append(scr)
        
    for k, v in mtr_all.items():
        print('{}: {:.3f}, {:.3f}'.format(k, np.mean(v), np.std(v)))       
    
        
    hmp_roc = get_roc_info(lst_lbl, lst_scr)
    hmp_pr  = get_pr_info(lst_lbl, lst_scr)
    
    fig, ax = plt.subplots(1)
    
    plot_roc('roc', ax, hmp_roc['xs'], hmp_roc['ys_mean'], hmp_roc['ys_upper'],
             hmp_roc['ys_lower'], hmp_roc['auc_mean'], hmp_roc['auc_std'])
    
    fig.savefig(f'{dir_rsl}/roc.png', dpi=200)
    
    fig, ax = plt.subplots(1)
    
    plot_roc('pr', ax, hmp_pr['xs'], hmp_pr['ys_mean'], hmp_pr['ys_upper'],
             hmp_pr['ys_lower'], hmp_pr['auc_mean'], hmp_pr['auc_std'])
    
    fig.savefig(f'{dir_rsl}/pr.png', dpi=200)
