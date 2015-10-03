import numpy as np
import nltk
from nltk.tokenize import wordpunct_tokenize

# Some constants. The weird order in the EMO list
# is to facilitate the coregionalization matrix plotting
EMOS = ['sadness','fear','anger','disgust','surprise','joy']
EMO_DICT = {}
EMO_DICT['anger'] = 0
EMO_DICT['disgust'] = 1
EMO_DICT['fear'] = 2
EMO_DICT['joy'] = 3
EMO_DICT['sadness'] = 4
EMO_DICT['surprise'] = 5


def preprocess_sent(sent, lemmatizer):
    """
    Take a sentence in string format and returns
    a list of lemmatized tokens.
    """
    tokenized = wordpunct_tokenize(sent.lower())
    sent = [lemmatizer.lemmatize(word) for word in tokenized]
    return sent


def read_sents_file(sents_file, train, test):
    """ 
    Read the sentences file and output two dicts,
    one containing the training set and another
    for the test set.
    "train" and "test" are the respective set sizes.
    We take the first "train" sentences as training set
    and the last "test" sentence as test set.
    For this reason "train" + "test must be < 1001,
    otherwise the sets will overlap.
    """
    data = np.loadtxt(sents_file, delimiter='_', dtype=object)
    train_sents = {}
    test_sents = {}
    data_size = data.shape[0]
    wnl = nltk.stem.WordNetLemmatizer()
    for i in xrange(data.shape[0]):
        sent = preprocess_sent(data[i][1], wnl)
        if i < train:
            train_sents[int(data[i][0])] = sent
        elif i >= (data_size - test):
            test_sents[int(data[i][0])] = sent
    return train_sents, test_sents


def build_word_dict(train_sents):
    """
    Return a dict with words mapped to their indices.
    Words that appear only on the test set will
    not appear here (and therefore won't count
    as features in the BOW model).
    """
    word_dict = {}
    for sent in train_sents.values():
        for word in sent:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict


def build_feat_vectors(train_sents, train_labels, test_sents,
                       test_labels, word_dict):
    """
    Build the feature vectors for train and test sets, 
    which encode a BOW representation for each sentence.
    """
    train_feats = np.zeros(shape=(len(train_sents), len(word_dict)))
    for i,row in enumerate(train_labels):
        sent_id = row[0]
        for word in train_sents[sent_id]:
            train_feats[i][word_dict[word]] += 1
    test_feats = np.zeros(shape=(len(test_sents), len(word_dict)))
    for i,row in enumerate(test_labels):
        sent_id = row[0]
        for word in test_sents[sent_id]:
            if word in word_dict:
                test_feats[i][word_dict[word]] += 1
    return train_feats, test_feats


def build_data(sents_file, labels_file, train, test):
    """
    Build and preprocess the data into a suitable format
    for the learning algorithms. The format is a dictionary,
    where the key is the emotion and the value is a matrix where
    the rows are the BOW vectors concatenated with the emotion label
    in the last column. It's very inefficient since it replicates
    the BOW vectors for each emotion but GPy needs that for the
    ICM models. Also, the dataset is small =P.
    """
    labels = np.loadtxt(labels_file)
    train_labels = labels[:train]
    test_labels = labels[-test:]
    train_sents, test_sents = read_sents_file(sents_file, train, test)
    word_dict = build_word_dict(train_sents)
    train_feats, test_feats = build_feat_vectors(train_sents, train_labels,
                                                 test_sents, test_labels,
                                                 word_dict)
    #train_data = {}
    #test_data = {}
    train_data = np.zeros(shape=(len(EMOS), train_feats.shape[0],
                                 train_feats.shape[1] + 1))
    test_data = np.zeros(shape=(len(EMOS), test_feats.shape[0],
                                test_feats.shape[1] + 1))
    for emo in EMOS:
        emo_id = EMO_DICT[emo] + 1
        train_data[emo_id - 1] = np.concatenate((train_feats, 
                                             train_labels[:,emo_id:emo_id+1]), axis=1)
        test_data[emo_id - 1] = np.concatenate((test_feats, 
                                            test_labels[:,emo_id:emo_id+1]), axis=1)
    return train_data, test_data
    

if __name__ == "__main__":
    import sys
    import scipy.io
    SENTS_FILE = sys.argv[1]
    LABELS_FILE = sys.argv[2]
    TRAIN = int(sys.argv[3])
    TEST = int(sys.argv[4])
    TRAIN_DUMP = sys.argv[5]
    TEST_DUMP = sys.argv[6]
    train_data, test_data = build_data(SENTS_FILE, LABELS_FILE, TRAIN, TEST)
    # Numpy savetxt does not work with 3D tensors so we use
    # Matlab files for that instead.
    scipy.io.savemat(TRAIN_DUMP, mdict={'out': train_data})
    scipy.io.savemat(TEST_DUMP, mdict={'out': test_data})

