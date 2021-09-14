"""
generate_saliency_vectors.py
generate saliency vectors;
"""
import torch
import numpy as np
from data import AudioDataset
from fhs_split_dataframe import has_transcript_and_mri
from model import Model
from tcn import TCN
from lstm_bi import LSTM
from map_saliency_to_transcripts import map_transcript
from misc import calc_performance_metrics
from sv_presets import get_sv_presets

def get_dset_tst(csv_info, **kwargs):
    """
    create test dataset;
    """
    return AudioDataset(csv_info, 'TST', **kwargs)

def get_model(model, device, n_concat):
    """
    get model;
    """
    if model == 'cnn':
        neural_network = TCN(device)
    elif model == 'lstm':
        neural_network = LSTM(13 * n_concat, 64, device)
    else:
        raise AssertionError(f'model type {model} is not supported;')
    return Model(n_concat=n_concat, device=device, nn=neural_network)

def get_vec(model):
    """
    get vec array;
    """
    for parameter in model.nn.mlp.parameters():
        vec = parameter.data[1,:] - parameter.data[0,:]
        return vec.view(1, 512, 1)
    return None

def create_final_vectors(scr, vec, dset_tst, stop_after=None):
    """
    create final saliency vectors;
    """
    final_vec_and_durations = []
    for idx, act in enumerate(scr):
        if stop_after is not None and idx >= stop_after:
            break
        final_vec = torch.mean(act * vec, dim=1).squeeze().data.cpu().numpy()
        mfcc_npy = dset_tst.audio_fns[idx]
        ## may need to check if this is accessing the right file;
        transcript = dset_tst.transcript_fns[idx]
        duration = int(len(np.load(mfcc_npy)) / 6000)
        ## length of mfcc_npy is length of audio in segments of 10 ms;
        ## * 10 gets us ms, / 1000 gets us to seconds;
        ## then dividing by 60 gets us to minutes;
        final_vec_and_durations.append((final_vec, np.load(mfcc_npy),
            duration, mfcc_npy, transcript))
    return final_vec_and_durations

def main():
    """
    main entrypoint;
    """
    csv_info, seed, pt_file = get_sv_presets()

    get_all_trn_test_kw = {'get_test_ids': has_transcript_and_mri}
    get_label = lambda r: int(str(r['is_ad']) == '1')
    tst_kw = {'tst_idx': None, 'seed': seed, 'holdout_test': True,
        'get_all_trn_test_kw': get_all_trn_test_kw, 'get_label': get_label}
    device = 1
    n_concat = 10
    model_name = 'cnn'

    dset_tst = get_dset_tst(csv_info, **tst_kw)
    ## test dataset;

    model_obj = get_model(model_name, device, n_concat)
    ## model object;

    model_obj.load_model(pt_file)
    ## load best model;

    prb = model_obj.prob(dset_tst, b_size=64)
    met = calc_performance_metrics(prb, dset_tst.labels)
    for metric, metric_val in met.items():
        if metric == "mat":
            continue
        print("\t{}, {:.3f}".format(metric, metric_val))
    ## calc performance and print;

    vec = get_vec(model_obj)
    ## linear layer parameters;

    scr = model_obj.eval_wo_gpool(dset_tst, b_size=64)
    ## activation vectors;

    final_vectors = create_final_vectors(scr, vec, dset_tst, stop_after=2)
    ## next step is to save the vectors from this list;
    for vector, mfcc, duration, mfcc_npy, transcript in final_vectors:
        print(vector.shape)
        print(mfcc.shape)
        print(duration)
        print(mfcc_npy)
        print(transcript)
        final = map_transcript(transcript, 16384)
        for sv_idx, time_dict in final.items():
            print(f'sv_idx: {sv_idx};')
            for np_test, np_dur_cs in time_dict.items():
                print(f'np_test: {np_test}; np_dur_seconds: {np_dur_cs / 100}')
        input()
    return final_vectors

if __name__ == "__main__":
    main()
