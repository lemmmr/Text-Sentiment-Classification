#Shell script that prepares the data for FastText and runs the code that generates the best submission file
#Command to run: ./run.sh
#Input files: 	data/train_pos_full.txt and data/train_neg_full.txt
#				data/test_data.txt		
#Final output:	final/FastText_ensemble_submission.csv

clear
mkdir proc
mkdir results
mkdir final

#This part of the code prepares the data to be processed by FastText by:
#		adding the appropriate labels
#		getting all characters to lowercase
#		deleting all double blank spaces
echo "Loading train_pos..."
cat data/train_pos.txt | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' | sed -e 's/^/__label__1 /g' > proc/positive_labeled_FT.txt
echo "Loading train_neg..."
cat data/train_neg.txt | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' | sed -e 's/^/__label__-1 /g' > proc/negative_labeled_FT.txt
echo "Loading test data..."
cat data/test_data.txt | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' | sed -e 's/^[^,]*,//' >  proc/test_data.txt

#This part of the code concatenates the labeled positive and negative datasets and shuffles them
echo "Concatenating and shuffling negative and positive Tweets for Fasttext..."
cat proc/positive_labeled_FT.txt proc/negative_labeled_FT.txt | perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@" > proc/mixed_pos_neg_labeled_FT.txt

#This part of the code runs the python file that produced our best submission
echo "Running fasttext..."
python fasttext200.py
echo "Done"




