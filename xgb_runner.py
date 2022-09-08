
import os
import argparse
from pathlib import Path
import sys
import logging

import numpy as np

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import sklearn.metrics as sklm

from parse_args import update_parser_train, update_args_json
from dataset import prepare_dataset, get_data
from timer import Timer


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

    parser.add_argument('--verbose', default='INFO', type=str,
                        choices=("ERROR", "WARNING", "INFO", "DEBUG"))
    return parser


def train(args, data):

    X_train, y_train = data.X_train, data.y_train
    
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

    with Timer() as t_train:
        model = model.fit(X_train, y_train)

    return model, t_train


def predict(model, data):

    X_test, y_test = data.X_test, data.y_test

    with Timer() as t_pred:
        prob_prediction = model.predict(X_test)

    pred_res = classification_metrics(y_test, prob_prediction)

    return pred_res, t_pred


def benchmark(args):
    
    data = prepare_dataset(args.datadir, args.dataset, args.nrows)

    booster, t_train = train(args, data)
    booster.save_model(f'xgb-{args.dataset}-model.json')

    pred_res, t_pred = predict(booster, data)

    print(f'xgb train time is : {t_train.interval}')
    print(f'xgb pred time is : {t_pred.interval}')

    print(pred_res)


def classification_metrics(y_true, y_prob, threshold=0.5):

    def evaluate_metrics(y_true, y_pred, metrics):
        res = {}
        for metric_name, metric in metrics.items():
            res[metric_name] = metric(y_true, y_pred)
        return res

    y_true = y_true.to_numpy()

    y_pred = np.where(y_prob > threshold, 1, 0)
    metrics = {
        "Accuracy": sklm.accuracy_score,
        "Log_Loss": lambda real, pred: sklm.log_loss(real, y_prob, eps=1e-5),
        # yes, I'm using y_prob here!
        "AUC": lambda real, pred: sklm.roc_auc_score(real, y_prob),
        "Precision": sklm.precision_score,
        "Recall": sklm.recall_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def main():

    parser = parse_args()
    update_parser_train(parser)

    args = parser.parse_args()
    
    logging.basicConfig(
    stream=sys.stdout, format='%(levelname)s: %(message)s', level=args.verbose)
    
    update_args_json(args, logging)

    # 
    benchmark(args)


if __name__ == '__main__':

    
    main()