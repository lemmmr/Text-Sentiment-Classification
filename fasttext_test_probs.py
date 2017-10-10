from helpers import *            
import fasttext
import numpy as np
import random

#Python script that performs one run of FastText
#Input files:   proc/mixed_pos_neg_labeled_FT.txt
#Output file:   results/FastText_output_probs.csv

#Path to training data (shuffled positive and negative datasets)
train_data_dir='proc/mixed_pos_neg_labeled_FT.txt'
#Path to test data 
test_data_dir='proc/test_data.txt'

#Defining FastText parameters:
epoch_val=5
lr_val=0.022
loss_val='ns'
ws_val=5
min_count_val=4
word_ngrams_val=3
thread_val=4
bucket_val=2000000

#Reading training data
with open(train_data_dir, encoding="utf-8") as d:
	all_tweets = d.readlines()

#Splitting the train data into train and test with a ratio of ratio_train in order to test the classifier locally
N = len(all_tweets)
ratio_train = 0.8
ind = random.sample(range(N),N)
ind_train=ind[1:int(N*ratio_train)+1]
ind_test = ind[int(N*ratio_train)+1:]
new_tweets = [all_tweets[e] for e in ind_train]
test_tweets =[all_tweets[e] for e in ind_test]


#Paths to collect splitted training data
train_data_dir_train='proc/mixed_pos_neg_train.txt'
train_data_dir_test='proc/mixed_pos_neg_test.txt'

#Creating the new files from splitting the train data
with open(train_data_dir_train, "wb") as f:
	for item in new_tweets:	
		 f.write(bytes(item,'UTF-8'))

with open(train_data_dir_test, "wb") as f:
	for item in test_tweets:	
		 f.write(bytes(item,'UTF-8'))

#Creating the classifier with FastText with the predifined parameters
classifier = fasttext.supervised(train_data_dir_train, 'results/model',epoch=epoch_val, lr=lr_val,loss=loss_val, 
    ws=ws_val,min_count=min_count_val,word_ngrams=word_ngrams_val,thread=thread_val,bucket=bucket_val)

#Reading the test data
fin=open(test_data_dir,'r')
test_data=fin.read().splitlines()
fin.close()

print("Testing model...\n")

#Getting the local evaluation of the classifier by using the splitted test data
results=classifier.test(train_data_dir_test)
print("Precision: %f" % results.precision)

#Getting the predictions on the test data as well as the probabilities
print("Getting the predictions...\n")
pred_labels=classifier.predict(test_data)
pred_probabilities=classifier.predict_proba(test_data)

#Counting how many predictions the classifier made
print("Counting how many predictions the classifier made:\n")
count_pos=0
count_neg=0
count_no_pred=0
list_no_pred=[]

for i in range(len(pred_labels)):
    #Counting positive tweets
    if pred_labels[i]==['1']:
        count_pos+=1
    #Counting negative tweets
    if pred_labels[i]==['-1']:
        count_neg+=1
    #Counting no predictions
    if pred_labels[i]==[]:
        count_no_pred+=1
        list_no_pred.append(i)

print("Positive tweets: %d" % count_pos)
print("Negative tweets: %d" % count_neg)
print("Total of predictions: %d\n" %(count_pos+count_neg))
print("Tweets without prediction: " )

#Getting the tweets that don't have a prediction
if count_no_pred>0:
	for i in range(len(list_no_pred)):
		print("%d: %s" %(list_no_pred[i]+1,test_data[list_no_pred[i]]))

#Using the probabilities to create a new list (to be used used in csv file)
pred_prods = np.zeros((len(pred_labels),1))
for ind, x in enumerate(pred_probabilities):
	for a in x:
		if int(a[0])==-1:
			pred_prods[ind]=a[1]
		elif int(a[0])==1:
			pred_prods[ind]=1-a[1]
		else:
			pred_prods[ind]=0
		
pred_prods=pred_prods.ravel().tolist()

#Generating lists for id for test data
id_labels=[]
for i in range(len(pred_labels)):
    id_labels.append(i+1)

#Generating file with probabilities to be averaged by fasttext200.py
print("Generating file with probabilities")
OUTPUT_PATH_probs = 'results/FastText_output_probs.csv'
create_csv_probs(id_labels, pred_prods, OUTPUT_PATH_probs)
print("Done")
