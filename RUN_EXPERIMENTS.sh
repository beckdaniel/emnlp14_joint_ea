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

runExperiments1() {

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
    # don't want to run.
    echo "RUNNING EXPERIMENTS 1..."
    mkdir -p plots

    echo "SVM..."
    mkdir -p results/svm
    #python scripts/svm.py $TRAIN_DATA $TEST_DATA results/svm
    echo ""

    echo "SINGLE GP..."
    mkdir -p results/single_gp
    #python scripts/single_gp.py $TRAIN_DATA $TEST_DATA results/single_gp
    echo ""

    #echo "ICM GP COMBINED..."
    #mkdir -p results/combined
    #python scripts/icm_gp.py $TRAIN_DATA $TEST_DATA results/combined combined
    #echo ""
    
    echo "ICM GP COMBINED+..."
    mkdir -p results/combined+
    #python scripts/icm_gp.py $TRAIN_DATA $TEST_DATA results/combined+ combined+
    echo ""

    for i in `seq 1`; do
	echo "ICM GP RANK $i..."
	mkdir -p results/rank_$i
	python scripts/icm_gp.py $TRAIN_DATA $TEST_DATA results/rank_$i plots rank $i
	echo ""
    done
};

# Comment here if you want to skip these.
runExperiments1

##########################
# 2) Data size experiments
##########################

runExperiments2() {

    echo "############################"
    echo "# 2) Performance experiments"
    echo "############################"
    echo ""

    TRAIN=900
    TEST=100
    TRAIN_DATA=data/exp2/train_data.mat
    TEST_DATA=data/exp2/test_data.mat

    # Preprocess dataset with the values we used in the
    # experiments, turning them into 3-D tensors 
    # (emotion, sentence, bow + label)
    echo "BUILDING TENSORS..."
    mkdir -p data/exp2
    if [[ ! -f data/exp2/train_data.mat ]]; then
    python scripts/preprocess.py $SENTS_FILE $LABELS_FILE $TRAIN $TEST \
	$TRAIN_DATA $TEST_DATA
    else
	echo "Tensors already built."
    fi
    echo ""

    echo "RUNNING EXPERIMENTS 2..."
    mkdir -p plots

    if [[! -f results/svm_size ]]; then
	echo "SVM..."
	mkdir -p results/svm_size
	python scripts/svm_size.py $TRAIN_DATA $TEST_DATA results/svm_size
	echo ""
    else
	echo "SVM results already calculated."
    fi

    if [[! -f results/single_gp_size ]]; then
	echo "SINGLE GP..."
	mkdir -p results/single_gp_size
	python scripts/single_gp_size.py $TRAIN_DATA $TEST_DATA results/single_gp_size
	echo ""
    else
	echo "Single GP results already calculated."
    fi

    if [[! -f results/rank_5_size ]]; then
	echo "ICM GP RANK 5"
	mkdir -p results/rank_5_size
	python scripts/icm_gp_size.py $TRAIN_DATA $TEST_DATA results/rank_5_size rank 5
	echo ""
    else
	echo "ICM GP rank 5 results already calculated."
    fi

};

# Comment here if you want to skip these.
runExperiments2


#############################
# 3) Score distribution plots
#############################
