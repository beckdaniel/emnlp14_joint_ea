#!/bin/bash

# Constants
ORIG_DATA_FILE=AffectiveText.Semeval.2007.tar.gz
DATA_DIR=AffectiveText.test
SENTS_FILE=$DATA_DIR/affectivetext.test.data
LABELS_FILE=$DATA_DIR/affectivetext_test.emotions.gold
TRAIN=100
TEST=100

# Prepare dataset
if [[ ! -f $DATADIR/affectivetext.test.data ]]; then
    # Extract the dataset, assuming it was already downloaded
    tar -xzf $ORIG_DATA_FILE
    # Change the xml to a normal text file
    head -1001 $DATA_DIR/affectivetext_test.xml | tail -1000 | \
	sed 's|^.*id="\([0-9]*\)">\(.*\)</instance>$|\1_\2|g' > $SENTS_FILE
fi

# Preprocess dataset with the values we used in the
# experiments

mkdir -p tmp_data
python scripts/preprocess.py $SENTS_FILE $LABELS_FILE $TRAIN $TEST \
    tmp_data/train_data.mat tmp_data/test_data.mat


