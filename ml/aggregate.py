# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

import ast
import argparse
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event
from pathlib import Path
FOLDER_NAME = 'aggregates'


def extract(dpath, subpath):
    scalar_accumulators = [EventAccumulator(os.path.join(dpath, dname, subpath)).Reload(
    ).scalars for dname in os.listdir(dpath) if dname != FOLDER_NAME]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, 'All runs need to have the same scalar keys'
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps
    all_steps = [tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events_per_key[0]]
    assert len(set(all_steps)) == 1, 'All runs need to have the same number of steps and the same step numbering'
    steps = all_steps[0]

    # Get and average wall times per step
    wall_times = np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events_per_key[0]], axis=0)

    values_per_key = {key: [[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for key, all_scalar_events in zip(keys, all_scalar_events_per_key)}

    return values_per_key, steps, wall_times


def write_summary(dpath, dname, fname, aggregations_per_key, steps, wall_times):
    fpath = os.path.join(dpath, dname)
    fpath = os.path.abspath(fpath)

    writer = tf.summary.FileWriter(fpath)

    for key, aggregations in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()

    # writer.close()



def write_csv(dpath, dname, fname, aggregations_per_key, steps, wall_times):
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    df = pd.DataFrame(np.transpose(list(aggregations_per_key.values())), index=steps, columns=aggregations_per_key.keys())
    df.to_csv(os.path.join(dpath, dname + '_' + fname + '.csv'), sep=';')


def aggregate(dpath, output, subpaths):
    name = os.path.basename(dpath)

    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std]

    write_ops = {
        'summary': write_summary,
        'csv': write_csv
    }

    print("Started aggregation {}".format(name))

    extracts_per_subpath = {subpath: extract(dpath, subpath) for subpath in subpaths}

    dpath = dpath.replace('./logs','./agglogs')
    for op in aggregation_ops:
        for subpath, (values_per_key, steps, wall_times) in extracts_per_subpath.items():
            path = os.path.join(dpath, FOLDER_NAME, subpath)

            dir = Path(os.path.join(dpath, FOLDER_NAME, subpath, op.__name__, name))
            dir.mkdir(parents=True, exist_ok=True)

            aggregations_per_key = {key: op(values, axis=0) for key, values in values_per_key.items()}
            write_ops.get(output)(path, op.__name__, name, aggregations_per_key, steps, wall_times)

    print("Ended aggregation {}".format(name))


if __name__ == '__main__':
    # def param_list(param):
    #     p_list = ast.literal_eval(param)
    #     if type(p_list) is not list:
    #         raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
    #     return p_list
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    # parser.add_argument("--subpaths", type=param_list, help="subpath sturctures", default=['test', 'train'])
    # parser.add_argument("--output", type=str, help="aggregation can be saves as tensorboard file (summary) or as table (csv)", default='summary')
    #
    # args = parser.parse_args()
    #
    # if not os.path.exists(args.path):
    #     raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(args.path))
    #
    # subpaths = [os.path.join(args.path, dname, subpath) for subpath in args.subpaths for dname in os.listdir(args.path)]
    #
    # for subpath in subpaths:
    #     if not os.path.exists(subpath):
    #         raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))
    #
    # if args.output not in ['summary', 'csv']:
    #     raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

# aggregate(args.path, args.output, args.subpaths)

    folder_path = "./logs/5channel_nolog_grid_norm_drop1_runstest/"

    exps = [os.path.join(folder_path, f) for f in os.listdir(folder_path) ]

    for exp in exps:
        aggregate(exp,'summary', ['test', 'train'])
