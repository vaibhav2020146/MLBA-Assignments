import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//train.csv')

# Load the test data
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//test.csv')

# Separate features and labels
X_train = train_data.drop(columns=['Labels'])
y_train = train_data['Labels']

# Drop the 'ID' column from the test data
X_test = test_data.drop(columns=['ID'])

y_test=test_data['ID']

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)

# Create a DataFrame for the predictions with 'ID' and 'Labels' columns
predictions = pd.DataFrame({'ID': y_test, 'Labels': y_pred})

# Save the predictions to a CSV file
predictions.to_csv('C://Users//91991//Desktop//MLBA//Assignment2//predictions_svm.csv', index=False)
