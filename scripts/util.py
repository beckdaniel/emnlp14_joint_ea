import numpy as np
from preprocess import EMOS

def print_results(maes, rmses, pearsons, all_pearson):
    """
    Pretty print the results from the experiments.
    """
    sorted_emos = sorted(EMOS)
    print '\t' + '\t'.join(sorted_emos) + ' avg'
    maes_to_print = [maes[emo] for emo in sorted_emos] + [np.mean(maes.values())]
    rmses_to_print = [rmses[emo] for emo in sorted_emos] + [np.mean(rmses.values())]
    ps_to_print = [pearsons[emo] for emo in sorted_emos] + [all_pearson]
    print 'MAE\t' + '\t'.join(['%.4f' % v for v in maes_to_print])
    print 'RMSE\t' + '\t'.join(['%.4f' % v for v in rmses_to_print])
    print 'Ps\t' + '\t'.join(['%.4f' % v for v in ps_to_print])
