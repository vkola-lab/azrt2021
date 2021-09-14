"""
map_saliency_to_transcripts.py
map saliency vector values to transcripts;
each saliency vector value represents 163.84 seconds;
"""
import csv
from collections import defaultdict

def yield_csv_data(csv_in):
    """
    parameters:
        csv_in(str): path to a csv file
    """
    with open(csv_in, newline='') as infile:
        for row in csv.DictReader(infile, delimiter=','):
            yield row

def ts_to_sec(hour, minute, second):
    """
    convert timestamp to seconds;
    """
    return hour * 60 * 60 + minute * 60 + second

def map_cs(start, end, sv_len_cs):
    """
    get indices of saliency vector values for a given row;
    """
    sv_idx_to_cs = defaultdict(int)
    for cs_ts in range(start, end):
        sv_idx_to_cs[cs_ts // sv_len_cs] += 1
    return sv_idx_to_cs

def map_transcript(csv_in, sv_len_cs):
    """
    map saliency values to transcript rows;
    """
    final = defaultdict(lambda: defaultdict(int))
    time_keys = ['hour', 'minute', 'second', 'duration']
    for row in yield_csv_data(csv_in):
        hour, minute, second, duration = [int(float(row[k])) for k in time_keys]
        start = ts_to_sec(hour, minute, second)
        end = start + duration
        start *= 100
        end *= 100
        ## convert seconds to centiseconds;
        ## start is inclusive, end is exclusive;
        ## [start, end)
        sv_idx_to_cs = map_cs(start, end, sv_len_cs)
        ## sv_idx_to_csv: {sv_idx: dur_in_cs, sv_idx+1: ##, ...}
        segment_name = row['segment_name']
        for sv_idx, total_cs in sv_idx_to_cs.items():
            current_time_dict = final[sv_idx]
            ## current_time_dict: {segment_name: ##}
            current_time_dict[segment_name] += total_cs
        # print(f'{hour}: {minute}: {second};')
        # print(sv_idx_to_cs)
    return final
