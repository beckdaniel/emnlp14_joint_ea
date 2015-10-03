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


def icm_gp_experiment(train_data, test_data, model, rank):
    maes = {}
    rmses = {}
    pearsons = {}
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []
    for emo in sorted(EMOS): # very important to sort here
        emo_id = EMO_DICT[emo]
        train_x = train_data[emo_id, :, :-1]
        train_y = train_data[emo_id, :, -1:]
        test_x = test_data[emo_id, :, :-1]
        test_y = test_data[emo_id, :, -1:]
        X_train_list.append(train_x)
        Y_train_list.append(train_y)
        X_test_list.append(test_x)
        Y_test_list.append(test_y)

    x_train, y_train, y_index = GPy.util.multioutput.build_XY(X_train_list, 
                                                              Y_train_list)
    Ny = 6
    k = GPy.util.multioutput.ICM(input_dim=x_train.shape[1]-1, num_outputs=Ny, 
                                 kernel=GPy.kern.RBF(x_train.shape[1]-1),
                                 W_rank=rank)
    m = GPy.models.GPRegression(x_train, y_train, kernel=k,
                                Y_metadata={'output_index': y_index})
    print m
    m.optimize_restarts(messages=False, max_iters=100)
    print m
    x_test, y_test, _ = GPy.util.multioutput.build_XY(X_test_list, Y_test_list)
    preds = m.predict(x_test, Y_metadata={'output_index': y_index})[0]
    factor = preds.shape[0] / 6
    all_labels = np.array([])
    all_preds = np.array([])
    
    for emo in EMOS:
        emo_id = EMO_DICT[emo]
        emo_preds = preds[emo_id * factor: (emo_id+1) * factor]
        emo_labels = y_test[emo_id * factor: (emo_id+1) * factor]
        maes[emo] = MAE(emo_preds, emo_labels)
        rmses[emo] = math.sqrt(MSE(emo_preds, emo_labels))
        pearsons[emo] = pearsonr(emo_preds, emo_labels)[0]
        all_labels = np.concatenate((all_labels, emo_labels.flatten()))
        all_preds = np.concatenate((all_preds, emo_preds.flatten()))
    all_pearson = pearsonr(all_preds, all_labels)[0]
    return maes, rmses, pearsons, all_pearson


if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    RESULTS_DIR = sys.argv[3]
    MODEL = sys.argv[4]
    if MODEL == "rank":
        RANK = int(sys.argv[5])
    else:
        RANK = 0
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    maes, rmses, pearsons, all_pearson = icm_gp_experiment(train_data, test_data,
                                                           MODEL, RANK)
    print_results(maes, rmses, pearsons, all_pearson)
    save_results(maes, rmses, pearsons, all_pearson, RESULTS_DIR)
    
    
