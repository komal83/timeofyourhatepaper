# timeofyourhatepaper
Data &amp; script for reproducibility of our paper

This repo contains all the data and scripts necessary to reproduce the results presented in our paper.

DATA

Test data are in the omonimous folder.
Haspeede+ training dataset is contained in fle "training_new.csv".

AlBERTo

The code contained in gpu_bert.ipynb is optimised  to be run on google colab with GPU as hardware accelerator.
It takes as input files the training data and it outputs a file containing all the columns of the training data plus the predicted labels. 
We run each simulation 5 times in order to smooth out any statitical fluctuations.
At each run the script saves the metrics from the python scikitlearn classification reports package in an external  google drive spreadsheet file. 
Ech seed was chosen randomly and manually inputed in the section "Constant". 
