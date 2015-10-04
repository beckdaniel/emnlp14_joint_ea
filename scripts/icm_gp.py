import numpy as np
import scipy.io
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr
from preprocess import EMOS, EMO_DICT
from util import save_results, print_results, plot_coreg_matrix
import sys
import math
import GPy
import os


def icm_gp_experiment(train_data, test_data, model, rank):
    maes = {}
    rmses = {}
    pearsons = {}
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []
    for emo_id, emo in enumerate(EMOS):
    #for emo in sorted(EMOS): # very important to sort here
        #emo_id = EMO_DICT[emo]
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
    if model == "combined" or model == "combined+":
        m['ICM.B.W'].constrain_fixed(1.0)
    if model == "combined":
        m['ICM.B.kappa'].tie_together()
    print m
    #m.optimize_restarts(messages=False, max_iters=100, robust=True)
    m.optimize(max_iters=100)
    print m
    W = m['ICM.B.W']
    kappa = m['ICM.B.kappa']
    B = W.dot(W.T) + np.diag(kappa)
    
    x_test, y_test, _ = GPy.util.multioutput.build_XY(X_test_list, Y_test_list)
    preds = m.predict(x_test, Y_metadata={'output_index': y_index})[0]
    factor = preds.shape[0] / 6
    all_labels = np.array([])
    all_preds = np.array([])
    
    for emo_id, emo in enumerate(EMOS):
        #emo_id = EMO_DICT[emo]
        emo_preds = preds[emo_id * factor: (emo_id+1) * factor]
        emo_labels = y_test[emo_id * factor: (emo_id+1) * factor]
        maes[emo] = MAE(emo_preds, emo_labels)
        rmses[emo] = math.sqrt(MSE(emo_preds, emo_labels))
        pearsons[emo] = pearsonr(emo_preds, emo_labels)[0]
        all_labels = np.concatenate((all_labels, emo_labels.flatten()))
        all_preds = np.concatenate((all_preds, emo_preds.flatten()))
    all_pearson = pearsonr(all_preds, all_labels)[0]
    return maes, rmses, pearsons, all_pearson, B


if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    RESULTS_DIR = sys.argv[3]
    PLOTS_DIR = sys.argv[4]
    MODEL = sys.argv[5]
    if MODEL == "rank":
        RANK = int(sys.argv[6])
    else:
        RANK = 1
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    maes, rmses, pearsons, all_pearson, B = icm_gp_experiment(train_data, test_data,
                                                              MODEL, RANK)
    print_results(maes, rmses, pearsons, all_pearson)
    save_results(maes, rmses, pearsons, all_pearson, RESULTS_DIR)
    if MODEL == "rank":
        plot_name = '_'.join([MODEL, str(RANK), 'matrix.png'])
    else:
        plot_name = '_'.join([MODEL, 'matrix.png'])
    plot_coreg_matrix(B, os.path.join(PLOTS_DIR, plot_name))
    
    
