import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys
import os
from util import EMOS
from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr

def plot_dist(labels, svm, icm, output_file):
    to_plot = []
    to_plot.append(sorted(labels))
    to_plot.append(sorted(svm))
    to_plot.append(sorted(icm))
    colors = ['b', 'r', 'y']

    fig, ax = plt.subplots()
    x = np.arange(0, 50)
    plt.ylim([0, 0.2])
    for p, c in zip(to_plot, colors):
        density = gaussian_kde(p)
        density.covariance_factor = lambda : .5
        density._compute_covariance()
        y = density(x)
        plt.plot(x, y, c=c)
        plt.fill_between(x, y, alpha=.3, antialiased=True, color=c)
    plt.legend(['Gold', 'SVM', 'ICM Rank 5'])
    plt.savefig(output_file)
    

if __name__ == "__main__":
    TEST_DATA = sys.argv[1]
    PREDS_DIR = sys.argv[2]
    PLOTS_DIR = sys.argv[3]
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    svm_preds = np.loadtxt(os.path.join(PREDS_DIR, 'svm.tsv'))
    icm_preds = np.loadtxt(os.path.join(PREDS_DIR, 'rank_5.tsv'))
    TEST_SIZE = test_data.shape[1]
    for i, emo in enumerate(EMOS):
        labels = test_data[i,:,-1]
        svm = svm_preds[i * TEST_SIZE: (i+1) * TEST_SIZE]
        icm = icm_preds[i * TEST_SIZE: (i+1) * TEST_SIZE]
        plot_dist(labels, svm, icm, os.path.join(PLOTS_DIR, emo + '.png'))
        
