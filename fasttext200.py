import os
import numpy as np
from helpers import * 

#Python script that runs FastText N=200 times and averages the results of the predictions
#Output file: results/FastText_ensemble_submission.csv
N=2

#Creating array to collect all the probabilities generated by each run of FastText
f_total = np.zeros((10000,1)).ravel()
#N runs of FastText
for i in range(N):
	os.system('clear')
	print('FastText Iteration',i+1)
	#Calling python file that generates probabilities for one run of FastText
	os.system('python fasttext_test_probs.py')
	#Reading created probabilities csv file
	f = np.genfromtxt('results/FastText_output_probs.csv', delimiter=",", skip_header=1)
	#Adding to previous sum of probabilities
	f_total = np.add(f[:,1],f_total)

#Averaging the probabilities over the N runs
pred_labels = f_total/N

#Generating lists for id of the test data
id_labels=[]
for i in range(len(pred_labels)):
    id_labels.append(i+1)

#Generating final submission file
os.system('clear')
print("Generating ensemble file for submission")
OUTPUT_PATH_submission = 'final/FastText_ensemble_submission.csv'
create_csv_submission_ensemble(id_labels, pred_labels, OUTPUT_PATH_submission)