#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:52:15 2020

@author: cxue2
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from scipy import interp

def calc_performance_metrics(scr, lbl):
    """
    calculate performance metrics;
    """
    met = dict()
    # prediction
    prd = (scr > .5) * 1
    # metrics
    met['mat'] = confusion_matrix(y_true=lbl, y_pred=prd)
    TN, FP, FN, TP = met['mat'].ravel()
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    sen = TP / (TP + FN)
    spc = TN / (TN + FP)
    met['acc'] = (TN + TP) / N
    met['balanced_acc'] = (sen + spc) / 2
    met['sen'] = sen
    met['spc'] = spc
    met['prc'] = TP / (TP + FP)
    met['f1s'] = 2 * (met['prc'] * met['sen']) / (met['prc'] + met['sen'])
    met['wt_f1s'] = f1_score(lbl, prd, average='weighted')
    met['mcc'] = (TP / N - S * P) / np.sqrt(P * S * (1-S) * (1-P))
    try:
        met['auc'] = roc_auc_score(y_true=lbl, y_score=scr)
    except KeyboardInterrupt as kbi:
        raise kbi
    except:
        met['auc'] = np.nan
    return met

def show_performance_metrics(met):
    """
    print performance metrics;
    """
    print('\tmat: {}'.format(np.array_repr(met['mat']).replace('\n', '')))
    print('\tacc: {}'.format(met['acc']))
    print('\tsen: {}'.format(met['sen']))
    print('\tspc: {}'.format(met['spc']))
    print('\tprc: {}'.format(met['prc']))
    print('\tf1s: {}'.format(met['f1s']))
    print('\tmcc: {}'.format(met['mcc']))
    print('\tauc: {}'.format(met['auc']))

def get_roc_info(lst_lbl, lst_scr):
    """
    calculate ROC information;
    """
    fpr_pt = np.linspace(0, 1, 1001)
    tprs, aucs = [], []
    for lbl, scr in zip(lst_lbl, lst_scr):
        fpr, tpr, _ = roc_curve(y_true=lbl, y_score=scr, drop_intermediate=True)
        tprs.append(interp(fpr_pt, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
    tprs_mean = np.mean(tprs, axis=0)
    tprs_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tprs_mean + tprs_std, 1)
    tprs_lower = np.maximum(tprs_mean - tprs_std, 0)
    auc_mean = auc(fpr_pt, tprs_mean)
    auc_std = np.std(aucs)
    auc_std = 1 - auc_mean if auc_mean + auc_std > 1 else auc_std
    rslt = {'xs': fpr_pt,
            'ys_mean': tprs_mean,
            'ys_upper': tprs_upper,
            'ys_lower': tprs_lower,
            'auc_mean': auc_mean,
            'auc_std': auc_std}

    return rslt

def pr_interp(rc_, rc, pr):
    """
    interpolate PR;
    """
    pr_ = np.zeros_like(rc_)
    locs = np.searchsorted(rc, rc_)
    for idx, loc in enumerate(locs):
        l = loc - 1
        r = loc
        r1 = rc[l] if l > -1 else 0
        r2 = rc[r] if r < len(rc) else 1
        p1 = pr[l] if l > -1 else 1
        p2 = pr[r] if r < len(rc) else 0

        t1 = (1 - p2) * r2 / p2 / (r2 - r1) if p2 * (r2 - r1) > 1e-16 else (1 - p2) * r2 / 1e-16
        t2 = (1 - p1) * r1 / p1 / (r2 - r1) if p1 * (r2 - r1) > 1e-16 else (1 - p1) * r1 / 1e-16
        t3 = (1 - p1) * r1 / p1 if p1 > 1e-16 else (1 - p1) * r1 / 1e-16

        a = 1 + t1 - t2
        b = t3 - t1 * r1 + t2 * r1
        pr_[idx] = rc_[idx] / (a * rc_[idx] + b)
    return pr_

def get_pr_info(lst_lbl, lst_scr):
    """
    calculate PR info;
    """
    rc_pt = np.linspace(0, 1, 1001)
    rc_pt[0] = 1e-16
    prs = []
    aps = []
    for lbl, scr in zip(lst_lbl, lst_scr):
        pr, rc, _ = precision_recall_curve(y_true=lbl, probas_pred=scr)
        aps.append(average_precision_score(y_true=lbl, y_score=scr))
        pr, rc = pr[::-1], rc[::-1]
        prs.append(pr_interp(rc_pt, rc, pr))

    prs_mean = np.mean(prs, axis=0)
    prs_std = np.std(prs, axis=0)
    prs_upper = np.minimum(prs_mean + prs_std, 1)
    prs_lower = np.maximum(prs_mean - prs_std, 0)
    aps_mean = np.mean(aps)
    aps_std = np.std(aps)
    aps_std = 1 - aps_mean if aps_mean + aps_std > 1 else aps_std
    rslt = {'xs': rc_pt,
            'ys_mean': prs_mean,
            'ys_upper': prs_upper,
            'ys_lower': prs_lower,
            'auc_mean': aps_mean,
            'auc_std': aps_std}

    return rslt
