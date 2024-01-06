                                                                                 GROUP-25
                                                                Protein Classification with LSTM and Keras 

This submission contains Python code for training a protein classification model using LSTM (Long Short-Term Memory) neural networks implemented with Keras. It preprocesses protein sequence data from CSV files, trains a model, and makes predictions on a test dataset.

Prerequisites:
1. Python 3.6 or higher
2. Libraries: pandas, numpy, scikit-learn, tensorflow (or tensorflow-gpu for GPU support)

Prepare your data:

1. Place your training dataset in a file named train.csv.
2. Place your test dataset in a file named test.csv.
3. For executing from command line set the location where your train.csv and test.csv files are present in your system in the python code.

Ensure that CSV files have the following columns as per the CSV files of training and testing given in the assignment:

1. ID: Unique identifier for each data point.
2. Sequence: The protein sequence data.
3. Label: The labels you want to predict (e.g., binary labels).

Run the model training and prediction script:
--> python protein_classification.py --train_file train.csv --test_file test.csv --submission_file submission.csv

This command will preprocess the data, train the model, make predictions on the test data, and save the results in a file named submission.csv.

Input and Output Files:
1. train.csv: Training dataset containing protein sequence data and labels.
2. test.csv: Test dataset containing protein sequence data.
3. submission.csv: Output file where the predicted labels for the test dataset will be saved.

Results:
After running the script, you will have a submission.csv file with the predicted labels for the test dataset. You can use this file to evaluate the model's performance and make submissions for protein classification tasks.