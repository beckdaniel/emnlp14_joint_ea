import numpy as np
import nltk
from nlkt.tokenize import wordpunct_tokenize


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
    data = np.loadtxt(sents_file, delimiter='\t', dtype=object)
    train_sents = {}
    test_sents = {}
    for i in xrange(data.shape[0]):
        sent = preprocess(data[i][1])
        if i < train:
            train_sents[int(data[i][0])] = sent
        elif i >= test:
            test_sents[int(data[i][0])] = sent
    return train_sents, test_sents


    
