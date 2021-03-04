"""
fhs_split_dataframe.py
functions meant to split the dataset into
training and validation sets;
"""
import random
import numpy as np

def get_all_trn_test(df_raw):
    """
    get list of all fhs_ids;
    get list of fhs ids that are in the static test fold;
    rest are the other_fhs_ids;
    """
    test_ids = get_fhs_ids(df_raw.loc[df_raw["duration_csv_out_list_len"] != "0"])
    all_ids = get_fhs_ids(df_raw)
    other_ids = np.array([fid for fid in all_ids if fid not in test_ids])
    return all_ids, test_ids, other_ids

def get_fhs_ids(df_pts):
    """
    from a dataframe, get idtype+id (forms a unique FHS ID);
    return an array with all the unique FHS IDs;
    """
    idtypes = df_pts.idtype.values.ravel('K')
    ids = df_pts.id.values.ravel('K')
    return np.unique([f'{idtypes[i]}-{str(ids[i]).zfill(4)}' for i, _ in enumerate(idtypes)])

def get_sample_ids(all_ids, other_ids, holdout_test):
    """
    raw_ids: all data sample participant IDs;
    other_ids: all participant IDs for samples that aren't test samples;
        only defined if holding test fold to be static (holdout_test is true);
    ## current dataframe contains all of the data samples if we're not
    ## holding the test set to be static;
    ## otherwise, if we are holding the test set to be static, use only df_other

    ## if holdout_test -> lst_p_all has all data samples that aren't in the static test set;
    ## if not holdout_test -> lst_p_all has all data samples;
    """
    return all_ids if not holdout_test else other_ids

def create_folds(sample_ids, num_folds, seed):
    """
    take datasamples, split them into a number of folds (num_folds), set random seed;
    """
    random.seed(seed)
    lst_idx = np.array(range(len(sample_ids)))
    random.shuffle(lst_idx)
    return [lst_idx[np.arange(len(sample_ids)) % num_folds == i] for i in range(num_folds)]

def get_fold(sample_ids, folds, vld_idx, tst_idx, mode):
    """
    fld: numpy array containing the folds and the data indices for that fofld;
    vld_idx: validation fold index;
    tst_idx: test fold index;
    mode: 'VLD', 'TST', 'TRN'
    """
    assert mode in {'TRN', 'VLD', 'TST'}, f"{mode} is not TRN VLD OR TST"
    if mode == 'VLD':
        idx = folds[vld_idx]
    elif mode == 'TST':
        idx = folds[tst_idx]
    elif mode == 'TRN':
        all_fold_indices = np.arange(len(folds))
        ## if 5 folds, then all_fold_indices = [0, 1, 2, 3, 4]
        all_fold_indices = all_fold_indices[all_fold_indices != vld_idx]
        all_fold_indices = all_fold_indices[all_fold_indices != tst_idx]
        ## keep all fold indices except for the TRN and VLD indices;
        idx = np.concatenate([folds[all_fold_indices[i]] for i in range(len(all_fold_indices))])
    return sample_ids[idx]

def get_holdout_fold(sample_ids, test_ids, folds, vld_idx, mode):
    """
    get fold if holdout_test is true;
    """
    assert mode in {'TRN', 'VLD', 'TST'}, f"{mode} is not TRN VLD OR TST"
    current_mode_ids = None
    if mode == 'TST':
        current_mode_ids = test_ids
    elif mode == 'TRN':
        all_fold_indices = np.arange(len(folds))
        all_fold_indices = all_fold_indices[all_fold_indices != vld_idx]
        idx = np.concatenate([folds[all_fold_indices[i]] for i in range(len(all_fold_indices))])
        current_mode_ids = sample_ids[idx]
    elif mode == 'VLD':
        idx = folds[vld_idx]
        current_mode_ids = sample_ids[idx]
    return current_mode_ids
