import csv

#Python script for csv submission creation
#The functions create_csv_probs and create_csv_submission_ensemble are called respectively by the scripts
#	fasttext_test_probs.py
#	fasttext200.py 

##############################
#fasttext_test_probs.py
##############################

def create_csv_probs(ids, y_pred, name):
#	Parameters:
#	ids: a list with the tweet ids
#	y_pred: a list with the probabilies of the prediction of each tweet
#	name: path to output file

#	Output: csv file with the probabilities of the prediction of each tweet for one run of FastText	
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            if r2!=[]:
                writer.writerow({'Id':int(r1),'Prediction':(r2)})
            else: #Tweets which were not classified, will have a positive value
                writer.writerow({'Id':int(r1),'Prediction':int(100)}) 

###############################
#fasttext200.py
###############################

def create_csv_submission_ensemble(ids, y_pred, name):
#	Parameters:
#	ids: a list with the tweet ids
#	y_pred: a list with the probabilies of the prediction of each tweet
#	name: path to output file

#	Output: csv file for submission which includes all the classification labels
#			1 for positive, -1 for negative
	with open(name, 'w') as csvfile:
		fieldnames = ['Id','Prediction']
		writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator='\n')
		writer.writeheader()
		for r1,r2 in zip(ids, y_pred):
			if (r2>0.5):
				writer.writerow({'Id':int(r1),'Prediction':int(-1)})
			else:
				writer.writerow({'Id':int(r1),'Prediction':int(1)})