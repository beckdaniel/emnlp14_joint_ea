#!/bin/bash

# Constants
ORIG_DATA_FILE=AffectiveText.Semeval.2007.tar.gz
DATA_DIR=AffectiveText.test
SENTS_FILE=$DATA_DIR/affectivetext.test.data
LABELS_FILE=$DATA_DIR/affectivetext_test.emotions.gold

# Prepare dataset
echo "PREPARING DATASET..."
if [[ ! -f $SENTS_FILE ]]; then
    # Extract the dataset, assuming it was already downloaded
    tar -xzf $ORIG_DATA_FILE
    # Change the xml to a normal text file
    head -1001 $DATA_DIR/affectivetext_test.xml | tail -1000 | \
	sed 's|^.*id="\([0-9]*\)">\(.*\)</instance>$|\1_\2|g' > $SENTS_FILE
else
    echo "Dataset already extracted."
fi
echo ""

############################
# 1) Performance experiments
############################

echo "############################"
echo "# 1) Performance experiments"
echo "############################"
echo ""

TRAIN=100
TEST=900
TRAIN_DATA=data/exp1/train_data.mat
TEST_DATA=data/exp1/test_data.mat

# Preprocess dataset with the values we used in the
# experiments, turning them into 3-D tensors 
# (emotion, sentence, bow + label)
echo "BUILDING TENSORS..."
mkdir -p data/exp1
if [[ ! -f data/exp1/train_data.mat ]]; then
    python scripts/preprocess.py $SENTS_FILE $LABELS_FILE $TRAIN $TEST \
	 $TRAIN_DATA $TEST_DATA
else
    echo "Tensors already built."
fi
echo ""

# Experiments are here, you can comment thes ones you
# don't want to run

#python scripts/svm.py $TRAIN_DATA $TEST_DATA
python scripts/single_gp.py $TRAIN_DATA $TEST_DATA
#python scripts/icm_gp.py combined
#python scripts/icm_gp.py combined+
#python scripts/icm_gp.py rank 1
#python scripts/icm_gp.py rank 2
#python scripts/icm_gp.py rank 3
#python scripts/icm_gp.py rank 4
#python scripts/icm_gp.py rank 5


##########################
# 2) Data size experiments
##########################

#############################
# 3) Score distribution plots
#############################
