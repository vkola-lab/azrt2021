# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:19:01 2018

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""
import os
import errno
import sys
import random
from datetime import datetime
from fhs_split_dataframe import segment_mfcc, has_transcript_and_mri
from handle_input import get_args
from select_task import select_task
from data import AudioDataset
from model import Model
from tcn import TCN
from lstm_bi import LSTM

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" #

def main():
    """
    main entrypoint;
    """
    args = get_args(sys.argv[1:])
    args = {k: v for k, v in args.items() if v is not None}
    task_csv_txt = args.get('task_csv_txt', 'task_csvs.txt')
    task_id = args.get('task_id', 0)
    csv_info, ext = select_task(task_id, task_csv_txt)

    get_label = None
    if '_vs_ad' in ext:
        get_label = lambda r: int(str(r['is_ad']) == '1')
    elif ext == 'norm_vs_mci':
        get_label = lambda r: int(str(r['is_mci']) == '1')

    model = args.get('model', 'cnn')
    if model.lower() not in ['cnn', 'lstm']:
        print(f'model type {model} is not supported;')
        sys.exit()
    ext = f'{model}_{ext}'
    device = int(args.get('device', 0))
    num_folds = int(args.get('num_folds', 5))
    holdout_test = args.get('holdout_test')
    test_transcript_mri = args.get('test_transcript_mri')
    debug_stop = args.get('debug_stop')
    no_save_model = args.get('no_save_model')
    negative_loss_weight = float(args.get('negative_loss_weight', 1))
    positive_loss_weight = float(args.get('positive_loss_weight', 1))
    weights = [negative_loss_weight, positive_loss_weight]
    sample_two_thirds = args.get('sample_two_thirds')
    lr = float(args.get('learning_rate', 1e-4))
    random_sampling_weight = args.get('random_sampling_weight')
    if random_sampling_weight is not None:
        random_sampling_weight = float(random_sampling_weight)
    do_segment_audio = args.get('do_segment_audio')
    audio_segment_min = int(args.get('audio_segment_min', 5))

    if do_segment_audio:
        segment_audio_kw = {'win_len_ms': 10, 'segment_length_min': audio_segment_min}
        get_row_data_kw = {'segment_audio': segment_mfcc, 'segment_audio_kw': segment_audio_kw}
    else:
        get_row_data_kw = {}

    no_write_fold_txt = args.get('no_write_fold_txt')

    n_epoch = int(args.get('n_epoch', 1))
    static_seeds = args.get('static_seeds')
    num_seeds = int(args.get('num_seeds', 1))
    replacement = True
    final_args = {'task_id': task_id, 'model': model, 'device': device, 'num_folds': num_folds,
        'holdout_test': holdout_test, 'debug_stop': debug_stop, 'no_save_model': no_save_model,
        'weights': weights, 'sample_two_thirds': sample_two_thirds,
        'do_segment_audio': do_segment_audio, 'audio_segment_min': audio_segment_min,
        'no_write_fold_txt': no_write_fold_txt, 'n_epoch': n_epoch, 'static_seeds': static_seeds,
        'num_seeds': num_seeds,
        'lr': lr, 'random_sampling_weight': random_sampling_weight, 'replacement': replacement}

    ext += "_github_test"

    if weights != []:
        w1, w2 = weights
        ext += f"_with_loss_weights_{w1}_{w2}"
    if random_sampling_weight is not None:
        ext += f'_with_random_sampling_weight_{random_sampling_weight}'
        if not replacement:
            ext += '_no_replacement'
    if sample_two_thirds:
        ext += "_two_thirds_sample_size"
    if holdout_test:
        ext += "_static_test_fold"
        if test_transcript_mri:
            ext += '_test_transcript_mri'
    if do_segment_audio:
        ext += f'_segment_audio_of_length_{audio_segment_min}'
    get_dir_rsl = lambda e, n, s: f'results/{e}/{n}_epochs/{s}'

    if static_seeds:
        if model == "lstm":
            # seed_list = [21269]
            # seed_list = [61962, 21269]
            # seed_list = [21269, 41840, 49405, 50034, 62607, 70160, 72687,
            #     73095, 74079, 9349, 96300]
            seed_list = [3934]
        else:
            seed_list = [65779]
    else:
        # seed_list = [21269, 19952]
        # seed_list = [21269]
        seed_list = [72901]
        for i in range(num_seeds):
            seed = random.randint(0, 100000)
            dir_rsl = get_dir_rsl(ext, n_epoch, seed)
            while os.path.isdir(dir_rsl):
                seed = random.randint(0, 100000)
                dir_rsl = get_dir_rsl(ext, n_epoch, seed)
            os.makedirs(dir_rsl)
            seed_list.append(seed)

    for seed in seed_list:
        dir_rsl = get_dir_rsl(ext, n_epoch, seed)
        time = str(datetime.now()).replace(' ', '_')
        dir_rsl = f'{dir_rsl}/{time}'
        assert not os.path.isdir(dir_rsl), dir_rsl
        # create folder for saving results
        try:
            os.makedirs(dir_rsl)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
        trn_dir = f'{dir_rsl}/trn'
        vld_dir = f'{dir_rsl}/vld'
        for dir_to_make in [trn_dir, vld_dir]:
            if not os.path.isdir(dir_to_make):
                os.makedirs(dir_to_make)
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
                   'holdout_test': holdout_test,
                   'get_row_data_kw': get_row_data_kw}
                if test_transcript_mri:
                    get_all_trn_test_func_kw = {'get_test_ids': has_transcript_and_mri}
                    kwargs['get_all_trn_test_kw'] = get_all_trn_test_func_kw
                if get_label is not None:
                    kwargs['get_label'] = get_label
                dset_trn = AudioDataset(csv_info, 'TRN', **kwargs)
                dset_vld = AudioDataset(csv_info, 'VLD', **kwargs)
                dset_tst = AudioDataset(csv_info, 'TST', **kwargs)

                # initialize model
                n_concat = 10
                if model == 'cnn':
                    nn = TCN(device)
                elif model == 'lstm':
                    nn = LSTM(13 * n_concat, 64, device)
                else:
                    raise AssertionError(f'model type {model} is not supported;')
                model_obj = Model(n_concat=n_concat, device=device, nn=nn)

                # train model
                model_fit_kw = {'dset_vld': dset_vld, 'n_epoch': n_epoch, 'b_size': 4,
                    'lr': lr, 'weights': weights, 'sample_two_thirds': sample_two_thirds,
                    'debug_stop': debug_stop, 'random_sampling_weight': random_sampling_weight,
                    'replacement': replacement}
                model_obj.fit(dset_trn, dir_rsl,**model_fit_kw)

                if not no_save_model:
                    if not os.path.isdir(f"pt_files/{ext}"):
                        os.makedirs(f"pt_files/{ext}")
                    model_obj.save_model(f"./pt_files/{ext}/"+\
                        f"{ext}_{i}_{seed}_{n_epoch}_{time}_epochs.pt")
                # evaluate model on validation dataset
                rsl = model_obj.prob(dset_tst, b_size=64)
                # break
                # save result to dataframe
                df_dat = dset_tst.df_dat
                df_dat['score'] = rsl

                # save dataframe to csv
                df_dat.to_csv('{}/audio_{}_{}.csv'.format(dir_rsl, seed, i), index=False)
                dset_trn.df_dat.to_csv(f'{trn_dir}/trn_audio_{seed}_{i}.csv', index=False)
                dset_vld.df_dat.to_csv(f'{vld_dir}/vld_audio_{seed}_{i}.csv', index=False)
                if not no_write_fold_txt:
                    txt_fp = os.path.join(dir_rsl, f"comb_[{i}].txt")
                    lines = []
                    for dset_ext, dataset in [('TRN',dset_trn), ('VLD', dset_vld), ('TST',
                            dset_tst)]:
                        line = f"{dset_ext}: num_patients: {dataset.num_patients}, num_audio: "+\
                            f"{dataset.num_audio} [negative_audio={dataset.num_negative_audio}, "+\
                            f"positive_audio={dataset.num_positive_audio}]\n"
                        lines.append(line)
                    with open(txt_fp, 'w') as outfile:
                        outfile.write(f'ext={ext}; seed={seed}; i={i};\n')
                        outfile.write("".join([f'{k}: {v}; ' for k, v in final_args.items()]) +\
                            "\n")
                        for line in lines:
                            outfile.write(line)
                        outfile.write(f"\nTRN IDs: {dset_trn.patient_list}\n\n")
                        outfile.write(f"VLD IDs: {dset_vld.patient_list}\n\n")
                        outfile.write(f"TST IDs: {dset_tst.patient_list}\n\n")
                # break

if __name__ == '__main__':
    main()
