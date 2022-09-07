
import os
import argparse
from pathlib import Path
import sys
import logging

import numpy as np

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from parse_args import update_parser_train, update_args_json


ROOT_PATH = Path(__file__).absolute()

def parse_args():
    
    parser = argparse.ArgumentParser(description='xgboost GBT benchmark')

    parser.add_argument('--configs', metavar='ConfigPath', type=str,
                        default='configs/config_example.json',
                        help='The path to a configuration file or '
                             'a directory that contains configuration files')

    parser.add_argument('--device', '--devices', default='host cpu gpu none', type=str, nargs='+',
                        choices=('host', 'cpu', 'gpu', 'none'),
                        help='Availible execution context devices. '
                        'This parameter only marks devices as available, '
                        'make sure to add the device to the config file '
                        'to run it on a specific device')

    parser.add_argument(
        "--dataset",
        default="all",
        type=str,
        help="The dataset to be used for benchmarking. 'all' for all datasets: "
        "fraud, epsilon, year, covtype, higgs, airline",
    )
    parser.add_argument(
        "--datadir", default=os.path.join(ROOT_PATH, "data"), type=str, help="The root datasets folder"
    )

    parser.add_argument('--seed', type=int, default=0,
                        help='Seed to pass as random_state')

    return parser


def train(args, X_train, y_train):

    model = XGBClassifier(
        learning_rate=args.learning_rate,
        gamma=args.min_split_loss,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        max_delta_step=args.max_delta_step,
        subsample=args.subsample,
        sampling_method='uniform',
        colsample_bytree=args.colsample_bytree,
        colsample_bylevel=1,
        colsample_bynode=1,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        tree_method=args.tree_method,
        scale_pos_weight=args.scale_pos_weight,
        grow_policy=args.grow_policy,
        max_leaves=args.max_leaves,
        max_bin=args.max_bin,
        objective=args.objective,
        seed=args.seed,
    )

    return model.fit(X_train, y_train)


def predict():

    pass


def benchmark():

    pass


def main():

    parser = parse_args()
    update_parser_train(parser)

    args = parser.parse_args()
    update_args_json(args, logging)

    logging.basicConfig(
    stream=sys.stdout, format='%(levelname)s: %(message)s', level=args['verbose'])


    pass

if __name__ == '__main__':

    
    main()