# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:19:01 2018

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""
import os
import errno
import sys
import random
from handle_input import cnn_argv
from cnn_model import Model_CNN
from data import AudioDataset

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" #

def main():
    """
    main entrypoint;
    """
    device = 2
    num_folds = 5
    holdout_test = False
    debug_stop = False
    save_model = True
    csv_info, ext, _ = cnn_argv(sys.argv)
    weights = [1, 2]
    sample_two_thirds = False
    write_fold_txt = True
    ext += "_github_test"

    if weights != []:
        w1, w2 = weights
        ext += f"_with_sampling_weights_{w1}_{w2}"
    if sample_two_thirds:
        ext += "_two_thirds_sample_size"
    if holdout_test:
        ext += "_static_test_fold"

    n_epoch = 1
    do_random = True
    num_seeds = 1
    get_dir_rsl = lambda e, n, s: f'results/cnn_{e}/{n}_epochs/{s}'

    if not do_random:
        seed_list = [21269, 19952]
    else:
        # seed_list = [21269, 19952]
        seed_list = []
        for i in range(num_seeds):
            seed = random.randint(0, 100000)
            dir_rsl = get_dir_rsl(ext, n_epoch, seed)
            while os.path.isdir(dir_rsl):
                seed = random.randint(0, 100000)
                dir_rsl = get_dir_rsl(ext, n_epoch, seed)
            seed_list.append(seed)

    seed_to_dir = {s: get_dir_rsl(ext, n_epoch, s) for s in seed_list}
    for seed, dir_rsl in seed_to_dir.items():
        assert not os.path.isdir(dir_rsl), dir_rsl
        # create folder for saving results
        try:
            os.makedirs(dir_rsl)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
        visited_outer_folds = []
        for i in range(num_folds):
            for j in range(num_folds):
                tst_idx = None if holdout_test else j
                ## tst fold is static if holdout_test is true;
                if tst_idx is None and i in visited_outer_folds:
                    ## only want to execute num_folds times, since tst fold is static;
                    continue
                if i == j:
                    continue ## vld and tst fold can't be the same
                visited_outer_folds.append(i)
                kwargs = {'num_folds': num_folds, 'vld_idx': i, 'tst_idx': tst_idx, 'seed': seed,
                   'holdout_test': holdout_test}
                dset_trn = AudioDataset(csv_info, 'TRN', **kwargs)
                dset_vld = AudioDataset(csv_info, 'VLD', **kwargs)
                dset_tst = AudioDataset(csv_info, 'TST', **kwargs)

                # initialize model
                model = Model_CNN(n_concat=10, device=device)

                # train model
                model_fit_kw = {'dset_vld': dset_vld, 'n_epoch': n_epoch, 'b_size': 4,
                    'lr': 1e-4, 'weights': weights, 'sample_two_thirds': sample_two_thirds,
                    'debug_stop': debug_stop}
                model.fit(dset_trn, **model_fit_kw)

                if save_model:
                    if not os.path.isdir(f"pt_files/{ext}"):
                        os.makedirs(f"pt_files/{ext}")
                    model.save_model(f"./pt_files/{ext}/"+\
                        f"{ext}_{i}_{seed}_{n_epoch}_epochs.pt")
                # evaluate model on validation dataset
                rsl = model.prob(dset_tst, b_size=64)
                # break
                # save result to dataframe
                df_dat = dset_tst.df_dat
                df_dat['score'] = rsl

                # save dataframe to csv
                df_dat.to_csv('{}/audio_{}_{}.csv'.format(dir_rsl, seed, i), index=False)
                if write_fold_txt:
                    txt_fp = os.path.join(dir_rsl, f"comb_[{i}].txt")
                    lines = []
                    for dset_ext, dataset in [('TRN',dset_trn), ('VLD', dset_vld), ('TST',
                            dset_tst)]:
                        line = f"{dset_ext}: num_patients: {dataset.num_patients}, num_audio: "+\
                            f"{dataset.num_audio} [negative_audio={dataset.num_negative_audio}, "+\
                            f"positive_audio={dataset.num_positive_audio}]\n"
                        lines.append(line)
                    with open(txt_fp, 'w') as outfile:
                        outfile.write(f"cnn; ext={ext}; seed={seed}; i={i}; "+\
                            f"n_epochs={n_epoch}; loss_weights={str(weights)};\n")
                        for line in lines:
                            outfile.write(line)
                        outfile.write(f"\nTRN IDs: {dset_trn.patient_list}\n\n")
                        outfile.write(f"VLD IDs: {dset_vld.patient_list}\n\n")
                        outfile.write(f"TST IDs: {dset_tst.patient_list}\n\n")
                # break

if __name__ == '__main__':
    main()
