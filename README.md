Authors: Yamane El-Zein, Camila Andrea González Williamson, Luis Emmanuel Medina Ríos.

Required installations:
	-Numpy: to install Numpy execute: pip install numpy
	-Fasttext: to install Fasttext execute: pip install fasttext

Folder structure should be as follows: 
	-data subdirectory: contains the following files
		-train_neg_full.txt: full train data set for negative tweets
		-train_pos_full.txt: full train data set for positive tweets
		-test_data.txt: test data set for predicting
	
	-Root directory:
		-run.sh: it is the bash script that has to be run to generate the submission file.
		-fasttext200.py: it is the python file called by run.sh that runs N=200 times the 		 fasttext_test_probs.py file and average all the probabilities of the predictions that 		 were generated
		-fasttext_test_probs.py: it is the python file that runs FastText once, called by 		 fasttext200.py
		-helpers.py: a python file that contains the functions to create the csv files.

The following directories are automatically created by running the run.sh script:
	-proc: contains the labeled files that were processed in order to use FastText.
		-mixed_pos_neg_labeled_FT.txt: final .txt file used to create the FastText model. it 		 contains both negative and positive labeled tweets.
		-mixed_pos_neg_test.txt: test data set created from splitting the train data set in 		 order to evaluate the model
		-mixed_pos_neg_train.txt: train data set created from splitting the train data set in 		 order to train the model
		-negative_labeled_FT.txt: a .txt file that contains the labeled negative tweets.
		-positive_labeled_FT.txt: a .txt file that contains the labeled positive tweets.
		-test_data.txt: a .txt with the test data without the tweet ids.
	-results: contains both the model (model.bin) created by FastText and the csv file with the 	 probabilities of one run of FastText:
		-FastText_output_probs.csv: Contains the probabilities of the predicted tweets.
		-model.bin: it is the model created by FastText.
	-final: contains the submission file
		-FastText_ensemble_submission.csv: the final submission file.

In order to get the submission file, one has to run the bash script (run.sh) and there should be a subdirectory in the root directory called “data” containing the files “train_neg_full.txt”, “train_pos_full.txt” and “test_data.txt”.
