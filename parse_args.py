
import os
import argparse
from pathlib import Path
import json
import logging

import utils


def update_parser_train(parser):

    parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                        help='Step size shrinkage used in update '
                            'to prevents overfitting')
    parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                        help='Minimum loss reduction required to make'
                            ' partition on a leaf node')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum depth of a tree')
    parser.add_argument('--min-child-weight', type=float, default=1,
                        help='Minimum sum of instance weight needed in a child')
    parser.add_argument('--max-delta-step', type=float, default=0,
                        help='Maximum delta step we allow each leaf output to be')
    parser.add_argument('--subsample', type=float, default=1,
                        help='Subsample ratio of the training instances')
    parser.add_argument('--colsample-bytree', type=float, default=1,
                        help='Subsample ratio of columns '
                            'when constructing each tree')
    parser.add_argument('--reg-lambda', type=float, default=1,
                        help='L2 regularization term on weights')
    parser.add_argument('--reg-alpha', type=float, default=0,
                        help='L1 regularization term on weights')
    parser.add_argument('--tree-method', type=str,
                        help='The tree construction algorithm used in XGBoost')
    parser.add_argument('--scale-pos-weight', type=float, default=1,
                        help='Controls a balance of positive and negative weights')
    parser.add_argument('--grow-policy', type=str, default='depthwise',
                        help='Controls a way new nodes are added to the tree')
    parser.add_argument('--max-leaves', type=int, default=0,
                        help='Maximum number of nodes to be added')
    parser.add_argument('--max-bin', type=int, default=256,
                        help='Maximum number of discrete bins to '
                            'bucket continuous features')
    parser.add_argument('--objective', type=str,
                        choices=('reg:squarederror', 'binary:logistic',
                                'multi:softmax', 'multi:softprob'),
                        help='Specifies the learning task')
    parser.add_argument('--backend', type=str,
                        choices=('skl', 'xgb'),
                        help='Specifies the API')

    parser.add_argument('--n-estimators', type=int, default=100,
                        help='The number of gradient boosted trees')

    parser.add_argument(
        "-nrows",
        default=None,
        type=int,
        help=(
            "Subset of rows in the datasets to use. Useful for test running "
            "benchmarks on small amounts of data. WARNING: Some datasets will "
            "give incorrect accuracy results if nrows is specified as they have "
            "predefined train/test splits."
        ),
    )


def update_args_json(args, logging):
    
    config_path = args.configs

    logging.info(f'Config: {config_path}')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    common_params = config['common']
    for params_set in config['cases']:
        params = common_params.copy()
        params.update(params_set.copy())
        # update arg dict
        vars(args).update(params)




