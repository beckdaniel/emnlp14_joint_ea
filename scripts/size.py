from svm import svm_experiment
import scipy

if __name__ == "__main__":
    TRAIN_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]
    RESULTS_DIR = sys.argv[3]
    train_data = scipy.io.loadmat(TRAIN_DATA)['out']
    test_data = scipy.io.loadmat(TEST_DATA)['out']
    all_pearsons = []
    for size in range(100, 601, 50):
        train_data_size = train_data[:, :size, :]
        maes, rmses, pearsons, all_pearson = svm_experiment(train_data_size, test_data)
        all_peasons.append(all_pearson)
    
    save_results(maes, rmses, pearsons, all_pearson, RESULTS_DIR)
    print_results(maes, rmses, pearsons, all_pearson)
