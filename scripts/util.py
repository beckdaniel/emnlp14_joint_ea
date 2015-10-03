import numpy as np
import os
from preprocess import EMOS


def save_results(maes, rmses, pearsons, all_pearson, results_dir):
    """
    Save results per emotion.
    """
    for emo in EMOS:
        result = [maes[emo], rmses[emo], pearsons[emo]]
        np.savetxt(os.path.join(results_dir, emo + '.tsv'), result)
    result = [np.mean(maes.values()), np.mean(rmses.values()), all_pearson]
    np.savetxt(os.path.join(results_dir, 'all.tsv'), result)
    

def print_results(maes, rmses, pearsons, all_pearson):
    """
    Pretty print the results from the experiments.
    """
    sorted_emos = sorted(EMOS)
    print '\t' + '\t'.join(sorted_emos) + ' all'
    maes_to_print = [maes[emo] for emo in sorted_emos] + [np.mean(maes.values())]
    rmses_to_print = [rmses[emo] for emo in sorted_emos] + [np.mean(rmses.values())]
    ps_to_print = [pearsons[emo] for emo in sorted_emos] + [all_pearson]
    print 'MAE\t' + '\t'.join(['%.4f' % v for v in maes_to_print])
    print 'RMSE\t' + '\t'.join(['%.4f' % v for v in rmses_to_print])
    print 'Pearson ' + '\t'.join(['%.4f' % v for v in ps_to_print])

