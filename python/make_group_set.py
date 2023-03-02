
import os
import argparse
from pathlib import Path
import sys
import json
import xgboost as xgb
import math

from analyse_tree import get_subtrees, make_group_set
from save_group_model import convert_tree_agg, save_tree_agg


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
    # model_path = "../xgb-higgs-model-1_6_1-ntrees_1k.json"
    # model_path = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep4_16.json"
    model_path = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_8_256full.json"

    if os.path.exists(model_path):
        booster_native = load_model('xgb', model_path)
        booster_sklearn = load_model('skl', model_path)
    else:
        print("model does not exist")

    # dump trees
    trees = booster_native.get_dump(dump_format="json")
    
    # load sub_trees
    tree_agg_num = 10
    tree_agg_list = []
    for i in range(math.ceil(len(trees)/tree_agg_num)):
        cur_idx = i * tree_agg_num
        groups_node = []
        groups_root = []
        for tree in trees[cur_idx : cur_idx + tree_agg_num]:
            sub_roots, sub_nodes = get_subtrees(json.loads(tree), 8, 3)
            groups_node.append(sub_nodes)
            groups_root.append(sub_roots)

        # make node share
        feat_set, gid_set = make_group_set(groups_node, cur_idx)

        # save
        groups_root = [x[0] for x in groups_root]
        tree_agg_dict = convert_tree_agg(groups_root, feat_set, gid_set)
        tree_agg_list.append(tree_agg_dict)

    save_tree_agg(tree_agg_list, 'test_full.json')







if __name__ == '__main__':

    main()