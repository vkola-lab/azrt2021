"""
test_model.py
module for testing using a model PT file (already trained);
"""
import os
from datetime import datetime
import fhs_split_dataframe as fsd
from data_from_csv import AudioDatasetFromCsv, segment_collate_fn
from model import Model
from tcn import TCN
from lstm_bi import LSTM

def test(model_obj, dset_tst, csv_out):
    """
    test a model pt file;

    model_obj: some Model
    dset_tst: AudioDatasetFromCsv with test set info
        csv_in: audio_<seed>_<fold_idx>.csv that resulted from original training of the model_obj
            should have patient_id, audio_fn, label, score
    csv_out: should have same CSV as csv_in, but adds a 'score_test' column that has the results
        from this round of testing;
    """
    results, rest_of_info_list = model_obj.prob(dset_tst, b_size=64, get_rest_of_info=True,
        eval_collate_fn=segment_collate_fn)
    start_list, end_list = [], []
    for _, _, this_start_list, this_end_list in rest_of_info_list:
        start_list.extend(this_start_list)
        end_list.extend(this_end_list)
    df_dat = dset_tst.df_dat
    df_dat['start'] = start_list
    df_dat['end'] = end_list
    df_dat['score_test'] = results

    df_dat.to_csv(csv_out, index=False)
    print(df_dat.columns)

def main():
    """
    main entrypoint
    2021-05-11;
    """
    ## cnn dir
    # parent_dir = "/encryptedfs/scripts/dvoice_lstm/final_azrt_github/azrt2021/azrt2021/"+\
    #     "model_pt_files_2021_05_11/norm_vs_demented_cnn_trained_full_audio/2021_05_11_seed_21269"

    ## cnn nondemented
    # parent_dir = "results/cnn_nondemented_vs_demented_github_test_with_loss_weights_1.0_1.0"+\
    #     "_with_random_sampling_weight_2.0_static_test_fold/64_epochs/3934/2021-05-22_01:26:59.563998"

    ## lstm nondemented
    parent_dir = "results/lstm_nondemented_vs_demented_github_test_with_loss_weights_1.0"+\
        "_1.0_with_random_sampling_weight_2.0_static_test_fold/32_epochs/3934/2021-05-22_18:13:16.470489"
    ## lstm dir
    # parent_dir = "results/lstm_norm_vs_demented_github_test_with_"+\
    #     "loss_weights_1.0_2.0_static_test_fold/8_epochs/21269/2021-05-11_11:06:04.065406"
    time = str(datetime.now()).replace(' ', '_')
    device = 2
    n_concat = 10
    # neural_network = TCN(device)
    neural_network = LSTM(13 * n_concat, 64, device)
    model_obj = Model(n_concat=n_concat, device=device, nn=neural_network)
    csv_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)\
        if f.lower().endswith('csv')]
    csv_files.sort()
    pt_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)\
        if f.lower().endswith('pt') and 'tmp' not in f.lower()]
    pt_files.sort()
    assert len(csv_files) == len(pt_files), f'{parent_dir}, {csv_files}, {pt_files}'
    for idx, pt_file in enumerate(pt_files):
        print(csv_files[idx])
        print(pt_file)
        print()
        print()
    input('check above...')
    for segment_length_min in [5, 10, 15]:
        for idx, model_pt_path in enumerate(pt_files):
            model_obj.load_model(model_pt_path)
            csv_in = csv_files[idx]
            segment_audio_kw = {'win_len_ms': 10, 'segment_length_min': segment_length_min,
                'do_return_array': True}
            kwargs = {'do_segment_audio': True, 'segment_audio': fsd.segment_mfcc,
                'segment_audio_kw': segment_audio_kw}
            dset_tst = AudioDatasetFromCsv(csv_in, **kwargs)
            base, ext = os.path.splitext(os.path.basename(csv_in))
            csv_out = f'{base}_{os.path.basename(model_pt_path)}{ext}'
            csv_out_parent = os.path.join(parent_dir, f'pt_files_aud_seg_{segment_length_min}_{time}')
            if not os.path.isdir(csv_out_parent):
                os.makedirs(csv_out_parent)
            csv_out = os.path.join(csv_out_parent, csv_out)
            # print(model_obj)
            # print(dset_tst)
            # print(csv_in)
            # print(csv_out)
            test(model_obj, dset_tst, csv_out)
            print(csv_out)

if __name__ == '__main__':
    main()
