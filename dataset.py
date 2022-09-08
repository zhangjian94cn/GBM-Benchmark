


import os
import sys
from enum import Enum
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class LearningTask(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3


class Data:
    def __init__(self, X_train, X_test, y_train, y_test, learning_task, qid_train=None, qid_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.learning_task = learning_task
        # For ranking task
        self.qid_train = qid_train
        self.qid_test = qid_test


def prepare_dataset(dataset_folder, dataset, nrows):
    dataset_folder = os.path.join(dataset_folder, dataset)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    prepare_function = globals()["prepare_" + dataset]
    return prepare_function(dataset_folder, nrows)



def prepare_fraud(dataset_folder, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    filename = "creditcard.csv"
    local_url = os.path.join(dataset_folder, filename)
    pickle_url = os.path.join(dataset_folder, "fraud" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    print("Preparing dataset ...")

    # os.system("kaggle datasets download mlg-ulb/creditcardfraud -f" + filename + " -p " + dataset_folder)
    df = pd.read_csv(local_url + ".zip", dtype=np.float32, nrows=nrows)
    X = df[[col for col in df.columns if col.startswith("V")]]
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)
    data = Data(
        X_train.astype("|f4").to_numpy(), X_test.astype("|f4").to_numpy(), y_train, y_test, LearningTask.CLASSIFICATION
    )
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_higgs(dataset_folder, nrows):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder, "higgs" + ("" if nrows is None else "-" + str(nrows)) + "-pickle.dat")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))
    print("Preparing dataset ...")

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    higgs = pd.read_csv(local_url, nrows=nrows, error_bad_lines=False)
    X = higgs.iloc[:, 1:]
    y = higgs.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=0.2,)
    data = Data(
        X_train.astype("|f4").to_numpy(), X_test.astype("|f4").to_numpy(), y_train, y_test, LearningTask.CLASSIFICATION
    )
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data



def get_data(data, size=-1):
    np_data = data.to_numpy() if not isinstance(data, np.ndarray) else data

    if size != -1:
        msg = "Requested size bigger than the data size (%d vs %d)" % (size, np_data.shape[0])
        assert size <= np_data.shape[0], msg
        np_data = np_data[0:size]

    return np_data





