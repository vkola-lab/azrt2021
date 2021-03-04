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
    parser.add_argument('-ti', '--task_id',
        help='input to select a task, which selects a csv, ext, and get_label;')
    parser.add_argument('-m', '--model',
        help='select a model (cnn or lstm), defaults to cnn;')
    parser.add_argument("-d", "--device", help='gpu device index;')
    parser.add_argument("-nf", "--num_folds", help='number of cross validation folds;')
    parser.add_argument('-ht', '--holdout_test', action='store_true',
        help='indicate whether or not to holdout a static test fold;')
    parser.add_argument('-db', '--debug_stop', action='store_false',
        help='indicate whether or not to debug and stop execution;')
    parser.add_argument('-sm', '--save_model', action='store_false',
        help='indicate whether or not to save the models;')
    parser.add_argument('-nlw', '--negative_loss_weight',
        help='loss weight for negative label;')
    parser.add_argument('-plw', '--positive_loss_weight',
        help='loss weight for positive label;')
    parser.add_argument('-stt', '--sample_two_thirds', action='store_true',
        help='indicates whether or not to sample only two thirds;')
    parser.add_argument('-wft', '--write_fold_txt', action='store_false',
        help='indicates whether or not to write the fold txt files;')
    parser.add_argument('-ne', '--n_epoch', help='indicates number of epochs;')
    parser.add_argument('-dr', '--do_random', action='store_false',
        help='indicates whether or not to pick random seed(s);')
    parser.add_argument('-ns', '--num_seeds', help='indicates number of seeds to use;')
    return parser.parse_args(args).__dict__