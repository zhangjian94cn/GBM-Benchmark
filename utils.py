import json
import os
import platform
import subprocess
import sys
import numpy as np

import sklearn.metrics as sklm
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, cast



def generate_cases(params: Dict[str, Union[List[Any], Any]]) -> List[str]:
    '''
    Generate cases for benchmarking by iterating the parameter values
    '''
    commands = ['']
    for param, values in params.items():
        if isinstance(values, list):
            prev_len = len(commands)
            commands *= len(values)
            dashes = '-' if len(param) == 1 else '--'
            for command_num in range(prev_len):
                for idx, val in enumerate(values):
                    commands[prev_len * idx + command_num] += ' ' + \
                        dashes + param + ' ' + str(val)
        else:
            dashes = '-' if len(param) == 1 else '--'
            for command_num, _ in enumerate(commands):
                commands[command_num] += ' ' + \
                    dashes + param + ' ' + str(values)
    return commands


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

