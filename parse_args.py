
import os
import argparse
from pathlib import Path


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

def parse_args_train(parser):

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
    parser.add_argument('--tree-method', type=str, required=True,
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
    parser.add_argument('--objective', type=str, required=True,
                        choices=('reg:squarederror', 'binary:logistic',
                                'multi:softmax', 'multi:softprob'),
                        help='Specifies the learning task')










