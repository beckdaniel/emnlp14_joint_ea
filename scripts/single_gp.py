import numpy as np
import scipy.io
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr
from preprocess import EMOS, EMO_DICT
from util import print_results, save_results
import sys
import math
import GPy
import os


def single_gp_experiment(train_data, test_data):
    maes = {}
    rmses = {}
    pearsons = {}
    all_labels = np.array([])
    all_preds = np.array([])
    for emo_id, emo in enumerate(EMOS):
        #emo_id = EMO_DICT[emo]
        train_x = train_data[emo_id, :, :-1]
        train_y = train_data[emo_id, :, -1:]
        test_x = test_data[emo_id, :, :-1]
        test_y = test_data[emo_id, :, -1:]
        
        k = GPy.kern.RBF(train_data[emo_id].shape[1] - 1)
        m = GPy.models.GPRegression(train_x, train_y, kernel=k)
        m.optimize_restarts(verbose=True, robust=True)
        preds = m.predict(test_x)[0]
        maes[emo] = MAE(preds, test_y)
        rmses[emo] = math.sqrt(MSE(preds, test_y))
        pearsons[emo] = pearsonr(preds, test_y)[0]
        all_labels = np.concatenate((all_labels, test_y.flatten()))
        all_preds = np.concatenate((all_preds, preds.flatten()))
    all_pearson = pearsonr(all_preds, all_labels)[0]
    return maes, rmses, pearsons, all_pearson, all_preds


if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    RESULTS_DIR = sys.argv[3]
    PREDS_DIR = sys.argv[4]
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    maes, rmses, pearsons, all_pearson, all_preds = single_gp_experiment(train_data,
                                                                         test_data)
    np.savetxt(os.path.join(PREDS_DIR, 'single_gp.tsv'), all_preds)
    save_results(maes, rmses, pearsons, all_pearson, RESULTS_DIR)
    print_results(maes, rmses, pearsons, all_pearson)
    
