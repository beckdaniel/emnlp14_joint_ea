import numpy as np
import scipy.io
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr
from preprocess import EMOS, EMO_DICT
from util import print_results
import sys
import math


def svm_experiment(train_data, test_data):
    maes = {}
    rmses = {}
    pearsons = {}
    hypers = {'C': np.logspace(-2, 2, 5),
              'epsilon': np.logspace(-3, 1, 5),
              'gamma': np.logspace(-3, 1, 5)}
    all_labels = np.array([])
    all_preds = np.array([])
    for emo in EMOS:
        emo_id = EMO_DICT[emo]
        m = GridSearchCV(SVR(), hypers)
        m.fit(train_data[emo_id, :, :-1], train_data[emo_id, :, -1])
        preds = m.predict(test_data[emo_id, :, :-1])
        maes[emo] = MAE(preds, test_data[emo_id, :, -1])
        rmses[emo] = math.sqrt(MSE(preds, test_data[emo_id, :, -1]))
        pearsons[emo] = pearsonr(preds, test_data[emo_id, :,-1])[0]
        all_labels = np.concatenate((all_labels, test_data[emo_id, :, -1]))
        all_preds = np.concatenate((all_preds, preds))
    all_pearson = pearsonr(all_preds, all_labels)[0]
    return maes, rmses, pearsons, all_pearson


if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    maes, rmses, pearsons, all_pearson = svm_experiment(train_data, test_data)
    print_results(maes, rmses, pearsons, all_pearson)
    
