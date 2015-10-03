# emnlp14_joint_ea
Code to reproduce experiments from the EMNLP 2014 paper: 
"Joint Emotion Analysis via Multi-task Gaussian Processes",
Daniel Beck, Trevor Cohn and Lucia Specia

https://aclweb.org/anthology/D/D14/D14-1190.pdf

To run the scripts, you need to first download the data
for the SEMEval-2007 "Affective Text" task:

https://web.eecs.umich.edu/~mihalcea/downloads.html#affective

You also need to install the following Python libraries (and their dependencies):

- Scikit-learn (pip install scikit-learn)
- GPy (pip install gpy)
- NLTK (pip install nltk)
- Inside NLTK, download the WordNet Corpus for the lemmatizer
  (python -c "import nltk; nltk.download()")

(PRO Tip: use a virtualenv for this =D)

After download the data and installing the libraries, two bash
scripts should be run:

- PREPROCESS_DATA.sh

This will untar the package and transform the data to a suitable
format for the experiments. Notice that we only use the "test"
folder.

- RUN_EXPERIMENTS.sh

This will run all the experiments and generate all the plots
reported in the paper (and some more). This takes a while.
If you wish to run only some of the experiments you can
comment the non-corresponding lines in this script.

If you find any problems, feel free to open an issue or
submit a pull request.

Daniel


