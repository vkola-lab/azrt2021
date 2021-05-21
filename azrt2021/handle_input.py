"""
handle_input.py
misc functions for giving input to CNN model running;
"""
import argparse

def get_args(args):
    """
    set variables based on cmd line args;
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-tct', '--task_csv_txt',
        help='path to the task csv txt file, defaults to task_csv.txt')
    parser.add_argument('-ti', '--task_id',
        help='input to select a task, which selects a csv, ext, and get_label;')
    parser.add_argument('-m', '--model',
        help='select a model (cnn or lstm), defaults to cnn;')
    parser.add_argument("-d", "--device", help='gpu device index;')
    parser.add_argument("-nf", "--num_folds", help='number of cross validation folds;')
    parser.add_argument('-ht', '--holdout_test', action='store_true',
        help='if set, test fold is held static;')
    parser.add_argument('-db', '--debug_stop', action='store_true',
        help='if set, execution is stopped short for debugging;')
    parser.add_argument('-nsm', '--no_save_model', action='store_true',
        help='if set, models will not be saved;')
    parser.add_argument('-nlw', '--negative_loss_weight',
        help='loss weight for negative label;')
    parser.add_argument('-plw', '--positive_loss_weight',
        help='loss weight for positive label;')
    parser.add_argument('-stt', '--sample_two_thirds', action='store_true',
        help='if set, only two thirds are randomly sampled;')
    parser.add_argument('-lr', '--learning_rate',
        help='assign the learning rate, default is 1e-4;')
    parser.add_argument('-rswt', '--random_sampling_weight',
        help='assign the positive to negative sampling weight ratio;')
    parser.add_argument('-dsa', '--do_segment_audio', action='store_true',
        help='if set, one segment of audio_segment_min minutes length is used as input data;')
    parser.add_argument('-asm', '--audio_segment_min',
        help='input to indicate length of audio segments, if using do_segment_audio;')
    parser.add_argument('-nwft', '--no_write_fold_txt', action='store_true',
        help='if set, fold txt files are not written;')
    parser.add_argument('-ne', '--n_epoch', help='indicates number of epochs;')
    parser.add_argument('-ss', '--static_seeds', action='store_true',
        help='if set, static seeds are used;')
    parser.add_argument('-ns', '--num_seeds', help='indicates number of seeds to use;')
    return parser.parse_args(args).__dict__
