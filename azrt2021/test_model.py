"""
test_model.py
module for testing using a model PT file (already trained);
"""
import os
import sys
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
    parent_dir = sys.argv[1]
    time = str(datetime.now()).replace(' ', '_')
    device = 2
    n_concat = 10
    if sys.argv[2].lower() == "lstm":
        neural_network = LSTM(13 * n_concat, 64, device)
    else:
        neural_network = TCN(device)
    test_full_aud = False
    if len(sys.argv) == 4:
        test_full_aud = int(sys.argv[3]) == 1

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
    if test_full_aud:
        print('testing full audio')
    else:
        print('testing segments 5/10/15')
    input('check above...')
    for idx, model_pt_path in enumerate(pt_files):
        csv_in = csv_files[idx]
        if not test_full_aud:
            test_segment(model_pt_path, model_obj, csv_in, parent_dir, time)
        else:
            test_full_audio(model_pt_path, model_obj, csv_in, parent_dir, time)

def test_segment(model_pt_path, model_obj, csv_in, parent_dir, time):
    """
    test pretrained model on segments of audio;
    """
    for segment_length_min in [5, 10, 15]:
        model_obj.load_model(model_pt_path)
        segment_audio_kw = {'win_len_ms': 10, 'segment_length_min': segment_length_min,
            'do_return_array': True}
        kwargs = {'do_segment_audio': True, 'segment_audio': fsd.segment_mfcc,
            'segment_audio_kw': segment_audio_kw}
        dset_tst = AudioDatasetFromCsv(csv_in, **kwargs)
        base, ext = os.path.splitext(os.path.basename(csv_in))
        csv_out = f'{base}_{os.path.basename(model_pt_path)}{ext}'
        csv_out_parent = os.path.join(parent_dir,
            f'pt_files_aud_seg_{segment_length_min}_{time}')
        if not os.path.isdir(csv_out_parent):
            os.makedirs(csv_out_parent)
        csv_out = os.path.join(csv_out_parent, csv_out)
        test(model_obj, dset_tst, csv_out)
        print(csv_out)

def test_full_audio(model_pt_path, model_obj, csv_in, parent_dir, time):
    """
    test full audio
    """
    model_obj.load_model(model_pt_path)
    dset_tst = AudioDatasetFromCsv(csv_in)
    base, ext = os.path.splitext(os.path.basename(csv_in))
    csv_out = f'{base}_{os.path.basename(model_pt_path)}{ext}'
    csv_out_parent = os.path.join(parent_dir,
        f'pt_files_full_{time}')
    if not os.path.isdir(csv_out_parent):
        os.makedirs(csv_out_parent)
    csv_out = os.path.join(csv_out_parent, csv_out)
    test(model_obj, dset_tst, csv_out)
    print(csv_out)

if __name__ == '__main__':
    main()
