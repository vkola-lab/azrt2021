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
    csv_info, ext, _ = cnn_argv(sys.argv)
    weights = [1, 2]
    sample_two_thirds = False
    ext += "_github_test"
    if weights != []:
        w1, w2 = weights
        ext += f"_with_sampling_weights_{w1}_{w2}"
    if sample_two_thirds:
        ext += "_two_thirds_sample_size"

    n_epoch = 2
    do_random = True
    get_dir_rsl = lambda e, n, s: f'results/holdout_test_{e}/{n}_epochs/{s}'

    if not do_random:
        seed_list = [21269, 19952]
    else:
        # seed_list = [21269, 19952]
        seed_list = []
        for i in range(10):
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

        for i in range(5):
            kwargs = {'comb': (i,), 'seed': seed, 'holdout_test': True}
            dset_trn = AudioDataset(csv_info, 'TRN', **kwargs)
            dset_vld = AudioDataset(csv_info, 'VLD', **kwargs)
            dset_tst = AudioDataset(csv_info, 'TST', **kwargs)

            # initialize model
            model = Model_CNN(n_concat=10, device=device)

            # train model
            model.fit(dset_trn, dset_vld=dset_vld, n_epoch=n_epoch, b_size=4, lr=1e-4,
                weights=weights, sample_two_thirds=sample_two_thirds)
            if not os.path.isdir(f"pt_files/{ext}"):
                os.makedirs(f"pt_files/{ext}")
            model.save_model(f"./pt_files/{ext}/{ext}_holdout_model_{i}_{seed}_{n_epoch}_epochs.pt")
            # evaluate model on validation dataset
            rsl = model.prob(dset_tst, b_size=64)
            # break
            # save result to dataframe
            df_dat = dset_tst.df_dat
            df_dat['score'] = rsl

            # save dataframe to csv
            df_dat.to_csv('{}/audio_{}_{}.csv'.format(dir_rsl, seed, i), index=False)
            txt_fp = os.path.join(dir_rsl, f"comb_[{i}].txt")
            lines = []
            for dset_ext, dataset in [('TRN',dset_trn), ('VLD', dset_vld), ('TST',
                    dset_tst)]:
                line = f"{dset_ext}: num_patients: {dataset.num_patients}, num_audio: "+\
                    f"{dataset.num_audio} [normal={dataset.num_normal_audio}, "+\
                    f"demented={dataset.num_demented_audio}]\n"
                lines.append(line)
            with open(txt_fp, 'w') as outfile:
                outfile.write(f"holdout_train; seed={seed}; i={i}; loss_weights={str(weights)};\n")
                for line in lines:
                    outfile.write(line)
                outfile.write(f"\nTRN IDs: {dset_trn.patient_list}\n\n")
                outfile.write(f"VLD IDs: {dset_vld.patient_list}\n\n")
                outfile.write(f"TST IDs: {dset_tst.patient_list}\n\n")
            # break

if __name__ == '__main__':
    main()
