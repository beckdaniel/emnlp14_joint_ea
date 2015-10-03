import numpy as np
import scipy.io
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr
from preprocess import EMOS, EMO_DICT
from util import print_results, save_results
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
        train_x = train_data[emo_id, :, :-1]
        train_y = train_data[emo_id, :, -1]
        test_x = test_data[emo_id, :, :-1]
        test_y = test_data[emo_id, :, -1]
        
        m = GridSearchCV(SVR(), hypers)
        m.fit(train_x, train_y)
        preds = m.predict(test_x)
        maes[emo] = MAE(preds, test_y)
        rmses[emo] = math.sqrt(MSE(preds, test_y))
        pearsons[emo] = pearsonr(preds, test_y)[0]
        all_labels = np.concatenate((all_labels, test_y))
        all_preds = np.concatenate((all_preds, preds))
    all_pearson = pearsonr(all_preds, all_labels)[0]
    return maes, rmses, pearsons, all_pearson


if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    RESULTS_DIR = sys.argv[3]
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    maes, rmses, pearsons, all_pearson = svm_experiment(train_data, test_data)
    save_results(maes, rmses, pearsons, all_pearson, RESULTS_DIR)
    print_results(maes, rmses, pearsons, all_pearson)
    
