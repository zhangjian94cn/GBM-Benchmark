
import os
from ast import arg
import argparse
from pathlib import Path
import sys
import logging
from timer import Timer
import numpy as np

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


import visualization
from parse_args import update_parser_train, update_parser_test, update_args_json
from dataset import prepare_dataset, get_data
from utils import classification_metrics


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

# 
def train(args, data, backend):

    def train_skl(args, data):

        X_train, y_train = data.X_train, data.y_train
        
        model = XGBClassifier(
            n_estimators=args.n_estimators,
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

        return model, t_train.interval

    def train_xgb(args, data):

        X_train, y_train = data.X_train, data.y_train
        dmatrix = xgb.DMatrix(X_train, y_train)
        
        param = vars(args)
        with Timer() as t_train:
            model = xgb.train(param, dmatrix, args.n_estimators)
        
        return model, t_train.interval

    func = f"train_{backend}"

    return eval(func)(args, data)

# predict group
def predict_baseline(args, model, data, backend):

    test_loop = args.test_loop

    def predict_skl(model, data):

        X_test, y_test = data.X_test, data.y_test

        with Timer() as t_pred:
            for i in range(test_loop):
                prob_prediction = model.predict(X_test)
            
        pred_res = classification_metrics(y_test, prob_prediction)

        return pred_res, t_pred.interval/args.test_loop

    def predict_xgb(model, data):

        X_test, y_test = data.X_test, data.y_test

        t_pred_sum = 0
        for i in range(test_loop):
            dmatrix = xgb.DMatrix(X_test, y_test)
            with Timer() as t_pred:
                prob_prediction = model.predict(dmatrix)
            
            t_pred_sum += t_pred.interval

        pred_res = classification_metrics(y_test, prob_prediction)

        return pred_res, t_pred_sum/args.test_loop

    func = f"predict_{backend}"
    
    return eval(func)(model, data)

def predict_hummingbird(args, model, data):
    import hummingbird.ml

    X_test, y_test = data.X_test, data.y_test
    
    print("Start model conversion")
    with Timer() as t_convert:
        hummingbird_model = hummingbird.ml.convert(model, "tvm", X_test)
    print(f"End model conversion. It costs {t_convert.interval}s")

    test_loop = args.test_loop
    with Timer() as t_pred:
        for i in range(test_loop):
            prob_prediction = hummingbird_model.predict(X_test)

    pred_res = classification_metrics(y_test, prob_prediction)
    
    return pred_res, t_pred.interval/args.test_loop

def predict_treelite(args, model, data):
    import treelite
    import treelite_runtime  

    # 
    X_test, y_test = data.X_test, data.y_test
    
    # 
    model = treelite.Model.from_xgboost(model)
    
    toolchain = 'gcc'
    print("Start model conversion")
    with Timer() as t_convert:
        # model.export_lib(toolchain=toolchain, libpath='./mymodel.so', verbose=True)
        model.export_lib(toolchain=toolchain, libpath='./mymodel.so',
                        params={'parallel_comp': 48}, verbose=True)
    print(f"End model conversion. It costs {t_convert.interval}s")

    predictor = treelite_runtime.Predictor('./mymodel.so', verbose=True)
    dmat = treelite_runtime.DMatrix(X_test)

    test_loop = args.test_loop
    with Timer() as t_pred:
        for i in range(test_loop):
            prob_prediction = predictor.predict(dmat)

    pred_res = classification_metrics(y_test, prob_prediction)
    return pred_res, t_pred.interval/args.test_loop

def predict_onedal(args, model, data):
    import daal4py as d4p

    X_test, y_test = data.X_test, data.y_test

    # Conversion to daal4py
    daal_model = d4p.get_gbt_model_from_xgboost(model)

    n_classes = len(np.unique(y_test))
    daal_predict_algo = d4p.gbt_classification_prediction(
        nClasses = n_classes,
        resultsToEvaluate="computeClassLabels|computeClassProbabilities",
        fptype='float'
    )

    test_loop = args.test_loop
    with Timer() as t_pred:
        for i in range(test_loop):
            prob_prediction = daal_predict_algo.compute(X_test, daal_model)

    pred_res = classification_metrics(y_test, prob_prediction.probabilities[:,1])
    return pred_res, t_pred.interval/args.test_loop

# 
def benchmark(args):

    data = prepare_dataset(args.datadir, args.dataset, args.nrows)

    # 0. model training
    # model_path = f"xgb-{args.dataset}-model.json"
    # model_path = "xgb-higgs-model-1_2_0.json"
    # model_path = "xgb-higgs-model-1_6_1.json"
    # model_path = "xgb-higgs-model-1_6_1-ntrees_1k.json"
    model_path = "xgb-higgs-model-1_5_0-ntrees_1k.json"
    # model_path = "xgb-higgs-model-0_90.json"

    # make the same model
    # model_path = "scikit-learn_bench/xgb-higgs1m-model.json"

    if os.path.exists(model_path):
        booster_native = load_model('xgb', model_path)
        booster_sklearn = load_model('skl', model_path)
    else:
        # booster_native, t_train_native = train(args, data, 'xgb')
        booster_sklearn, t_train_sklearn = train(args, data, 'skl')
        booster_sklearn.save_model(model_path)
        booster_native = load_model('xgb', model_path)

        print(f'xgb train time is : {t_train_sklearn}')

    # 1. xgboost as baseline
    pred_res, t_pred_native = predict_baseline(args, booster_native, data, 'xgb')
    pred_res, t_pred_sklearn = predict_baseline(args, booster_sklearn, data, 'skl')
    print(f'xgb native API pred time is : {t_pred_native}')
    print(f'xgb sklearn API pred time is : {t_pred_sklearn}')
    print(f'xgb result: {pred_res}')

    if args.visualize:
        visualization.visualize(booster_native, args.hist_path)

    # # 2. test hummingbird
    # pred_res_hb, t_pred_hb = predict_hummingbird(args, booster_sklearn, data)
    # print(f'hbird pred time is : {t_pred_hb}')
    # print('hbird result: ', pred_res_hb)

    # # 3. test treelite
    # pred_res_tl, t_pred_tl = predict_treelite(args, booster_native, data)
    # print(f'treelite pred time is : {t_pred_tl}')
    # print('treelite result: ', pred_res_tl)
    
    # 4. test onedal 
    pred_res_od, t_pred_od = predict_onedal(args, booster_native, data)
    print(f'oneDAL pred time is : {t_pred_od}')
    print('oneDAL result: ', pred_res_od)

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

    parser = parse_args()
    update_parser_train(parser)
    update_parser_test(parser)

    args = parser.parse_args()
    
    logging.basicConfig(
    stream=sys.stdout, format='%(levelname)s: %(message)s', level=args.verbose)
    
    update_args_json(args, logging)

    # 
    benchmark(args)


if __name__ == '__main__':

    main()