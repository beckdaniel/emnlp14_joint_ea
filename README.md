# emnlp14_joint_ea
Code to reproduce experiments from the EMNLP 2014 paper: 
"Joint Emotion Analysis via Multi-task Gaussian Processes",
Daniel Beck, Trevor Cohn and Lucia Specia

https://aclweb.org/anthology/D/D14/D14-1190.pdf

# Instructions

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

After download the data and installing the libraries, you can
run the following bash script

- RUN_EXPERIMENTS.sh

This will run all the experiments and generate all the plots
reported in the paper (and some more). This takes a while.
If you wish to run only some of the experiments you can
comment the corresponding lines at the end of the script.

# Analysing results

After running everything, all results will be at the "results" folder.
The equivalent to Table 1 is the file "final_pearsons.tsv", which is in
a friendly to read format. 
The internal folders store some additional metrics (MAE, RMSE) in a readable 
but non-friendly format.

The plots folder contains plots for the coregionalization matrices and 
the score distributions plots.

# Important remarks

The results shown in this experiment do not match exactly the ones in the paper.
Specifically, the SVM baseline performs better because it uses a larger grid in
grid search. The GP models also resulted in slightly different results but mainly
due to changes in the GPy implementation.

# TODO list

- The "combined" model is not here because the latest GPy version does not
accept parameter tying (yet).
- The dataset size experiments (section 4.4) are there but the plotting procedure
is missing.

If you find any problems, feel free to open an issue or
submit a pull request.

Daniel


