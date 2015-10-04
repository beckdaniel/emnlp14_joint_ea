from svm import svm_experiment
from single_gp import single_gp_experiment
from icm_gp import icm_gp_experiment
import scipy
import numpy as np
import os
import sys

if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    RESULTS_DIR = sys.argv[3]
    MODEL = sys.argv[4]
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    all_pearsons = []
    for size in range(100, 151, 50):
        train_data_size = train_data[:, :size, :]
        print str(size) + " ",
        if MODEL == "svm":
            _, _, _, all_pearson = svm_experiment(train_data_size, test_data)
        elif MODEL == "single_gp":
            _, _, _, all_pearson = single_gp_experiment(train_data_size, test_data)
        elif MODEL == "icm_gp":
            _, _, _, all_pearson, _ = icm_gp_experiment(train_data_size, test_data,
                                                        "rank", 5)                
        all_pearsons.append(all_pearson)
    print ""
    np.savetxt(os.path.join(RESULTS_DIR, MODEL + '_size.tsv'),
               np.array(all_pearsons))
