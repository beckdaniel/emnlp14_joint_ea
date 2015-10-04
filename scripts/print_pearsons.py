import sys
import os
import numpy as np
from util import EMOS


def print_row(row):
    print '\t'.join(["%10s" % i for i in row])
    

if __name__ == "__main__":
    RESULTS_DIR = sys.argv[1]
    models_dirs = os.listdir(RESULTS_DIR)
    if 'final_pearsons.tsv' in models_dirs:
        models_dirs.remove('final_pearsons.tsv')
    pearsons = {}
    pearsons['title'] = ['MODEL'] + sorted(EMOS) + ['all']
    for model in models_dirs:
        pearsons[model] = [model]
        for emo in sorted(EMOS) + ['all']:
            results_file = os.path.join(RESULTS_DIR, model, emo + '.tsv')
            results = np.loadtxt(results_file)
            pearsons[model].append("%.4f" % results[2])

    print_row(pearsons['title'])
    print_row(pearsons['svm'])
    print_row(pearsons['single_gp'])
    print_row(pearsons['combined+'])
    for i in range(1, 6):
        print_row(pearsons['rank_' + str(i)])
        
