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
    mkdir -p results
    mkdir -p preds
    mkdir -p plots/matrices

    if [[ ! -f preds/svm.tsv ]]; then
	echo "SVM..."
	mkdir -p results/svm
	python scripts/svm.py $TRAIN_DATA $TEST_DATA results/svm preds
	echo ""
    else
	echo "SVM results already calculated."
    fi

    if [[ ! -f preds/single_gp.tsv ]]; then
	echo "SINGLE GP..."
	mkdir -p results/single_gp
	python scripts/single_gp.py $TRAIN_DATA $TEST_DATA results/single_gp preds
	echo ""
    else
	echo "Single GP results already calculated."
    fi

    #echo "ICM GP COMBINED..."
    #mkdir -p results/combined
    #python scripts/icm_gp.py $TRAIN_DATA $TEST_DATA results/combined combined
    #echo ""
    
    if [[ ! -f preds/combined+.tsv ]]; then
	echo "ICM GP COMBINED+..."
	mkdir -p results/combined+
	python scripts/icm_gp.py $TRAIN_DATA $TEST_DATA results/combined+ preds plots/matrices combined+
	echo ""
    else
	echo "ICM GP COMBINED+ results already calculated."
    fi

    for i in `seq 5`; do
	if [[ ! -f preds/rank_$i.tsv ]]; then
	    echo "ICM GP RANK $i..."
	    mkdir -p results/rank_$i
	    python scripts/icm_gp.py $TRAIN_DATA $TEST_DATA results/rank_$i preds plots/matrices rank $i
	    echo ""
	else
	    echo "ICM GP RANK $i results already calculated."
	fi
    done

    # Wrap final pearson results and print them in a friendly format
    python scripts/print_pearsons.py results > results/final_pearsons.tsv

};

##########################
# 2) Data size experiments
##########################

runExperiments2() {

    echo "############################"
    echo "# 2) Data size experiments"
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
    mkdir -p results

    if [[ ! -f results/svm_size.tsv ]]; then
	echo "SVM..."
	python scripts/size.py $TRAIN_DATA $TEST_DATA results svm
	echo ""
    else
	echo "SVM results already calculated."
    fi

    if [[ ! -f results/single_gp_size.tsv ]]; then
	echo "SINGLE GP..."
	python scripts/size.py $TRAIN_DATA $TEST_DATA results single_gp
	echo ""
    else
	echo "Single GP results already calculated."
    fi

    if [[ ! -f results/icm_gp_size.tsv ]]; then
	echo "ICM GP RANK 5"
	python scripts/size.py $TRAIN_DATA $TEST_DATA results icm_gp rank 5
	echo ""
    else
	echo "ICM GP rank 5 results already calculated."
    fi

};

#############################
# 3) Score distribution plots
#############################

runExperiments3() {
    if [[ ! -f preds/single_gp.tsv ]]; then
	echo "Need to run Experiment 1 first"
    else
	echo "Plotting score distributions"

	mkdir -p plots/dists
	TEST_DATA=data/exp1/test_data.mat
	python scripts/plot_dists.py $TEST_DATA preds plots/dists
    fi
	
}


# Comment/uncomment as you like. Be aware that runExperiment2
# can take quite a long time.

runExperiments1
#runExperiments2
runExperiments3