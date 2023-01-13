
import os
import argparse
from pathlib import Path
import sys
import json
import xgboost as xgb

from analyse_tree import get_subtree, make_group_set

ROOT_PATH = Path(__file__).absolute()

def parse_args():
    
    parser = argparse.ArgumentParser(description='xgboost GBT benchmark')

    parser.add_argument('--configs', metavar='ConfigPath', type=str,
                        default='configs/config_example.json',
                        help='The path to a configuration file or '
                             'a directory that contains configuration files')
    return parser


def load_model(backend, model_path):

    if backend == 'xgb':
        booster = xgb.Booster()
        booster.load_model(model_path)

    elif backend == 'skl':
        booster = xgb.XGBClassifier()
        booster.load_model(model_path)

    else:
        raise("error backend")

    return booster


def main():

    # 0. model training
    # model_path = "../xgb-higgs-model-1_6_1-ntrees_1k_8_256full.json"
    model_path = "../xgb-higgs-model-1_6_1-ntrees_1k.json"
    # model_path = "../xgb-higgs-model-1_6_1-ntrees_1k_4_16.json"

    

    if os.path.exists(model_path):
        booster_native = load_model('xgb', model_path)
        booster_sklearn = load_model('skl', model_path)
    else:
        print("model does not exist")

    # 
    trees = booster_native.get_dump(dump_format="json")
    
    subtrees = []
    for tree in trees:
        subtrees.append(get_subtree(json.loads(tree), 4))

    make_group_set(subtrees)
    # 
    # print(subtrees)
    # visualization.distribution(booster_native)

    # if 0:
        # visualization.distribution(booster_native)
        # visualization.visualize(booster_native, args.hist_path)
        # visualization.plot_tree(booster_native)



if __name__ == '__main__':

    main()