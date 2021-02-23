"""
fhs_split_dataframe.py
functions meant to split the dataset into
training and validation sets;
"""
import random
import numpy as np

def get_trn_test(df_raw):
    """
    get dataframe with all training participants;
    get dataframe with all test participants;
    """
    df_test = df_raw.loc[df_raw["duration_csv_out_list_len"] != "0"]
    df_other = df_raw.loc[df_raw["duration_csv_out_list_len"] == "0"]
    return df_test, df_other

def get_fhs_ids(df_pts):
    """
    from a dataframe, get idtype+id (forms a unique FHS ID);
    return an array with all the unique FHS IDs;
    """
    idtypes = df_pts.idtype.values.ravel('K')
    ids = df_pts.id.values.ravel('K')
    return np.unique([f'{idtypes[i]}-{str(ids[i]).zfill(4)}' for i, _ in enumerate(idtypes)])

def get_samples(df_raw, df_other, holdout_test):
    """
    df_raw: dataframe with all data samples;
    df_other: dataframe with all samples that aren't test samples;
        only defined if holding test fold to be static (holdout_test is true);
    """
    df_pts = df_raw if not holdout_test else df_other
    ## current dataframe contains all of the data samples if we're not
    ## holding the test set to be static;
    ## otherwise, if we are holding the test set to be static, use only df_other
    lst_p_all = get_fhs_ids(df_pts)
    ## if holdout_test -> lst_p_all has all data samples that aren't in the static test set;
    ## if not holdout_test -> lst_p_all has all data samples;
    return lst_p_all

def create_folds(lst_p_all, num_folds, seed):
    """
    take datasamples, split them into a number of folds (num_folds), set random seed;
    """
    random.seed(seed)
    lst_idx = np.array(range(len(lst_p_all)))
    random.shuffle(lst_idx)
    return [lst_idx[np.arange(len(lst_p_all)) % num_folds == i] for i in range(num_folds)]

def get_fold(lst_p_all, fld, vld_idx, tst_idx, mode):
    """
    fld: numpy array containing the folds and the data indices for that fofld;
    vld_idx: validation fold index;
    tst_idx: test fold index;
    mode: 'VLD', 'TST', 'TRN'
    """
    assert mode in {'TRN', 'VLD', 'TST'}, f"{mode} is not TRN VLD OR TST"
    if mode == 'VLD':
        idx = fld[vld_idx]
    elif mode == 'TST':
        idx = fld[tst_idx]
    elif mode == 'TRN':
        all_fold_indices = np.arange(len(fld))
        ## if 5 folds, then all_fold_indices = [0, 1, 2, 3, 4]
        all_fold_indices = all_fold_indices[all_fold_indices != vld_idx]
        all_fold_indices = all_fold_indices[all_fold_indices != tst_idx]
        ## keep all fold indices except for the TRN and VLD indices;
        idx = np.concatenate([fld[all_fold_indices[i]] for i in range(len(all_fold_indices) - 2)])
    return lst_p_all[idx]

def get_holdout_fold(lst_p_all, df_test, fld, vld_idx, mode):
    """
    get fold if holdout_test is true;
    """
    assert mode in {'TRN', 'VLD', 'TST'}, f"{mode} is not TRN VLD OR TST"
    lst_p = None
    if mode == 'TST':
        lst_p = get_fhs_ids(df_test)
    elif mode == 'TRN':
        all_fold_indices = np.arange(len(fld))
        all_fold_indices = all_fold_indices[all_fold_indices != vld_idx]
        idx = np.concatenate([fld[all_fold_indices[i]] for i in range(len(all_fold_indices) - 1)])
        lst_p = lst_p_all[idx]
    elif mode == 'VLD':
        idx = fld[vld_idx]
        lst_p = lst_p_all[idx]
    return lst_p
