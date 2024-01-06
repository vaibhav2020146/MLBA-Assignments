import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//train.csv')

# Load the test data
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//test.csv')

# Extract features and labels from the training data
X_train = train_data.drop(columns=['Labels'])
y_train = train_data['Labels']

# Extract features from the test data
X_test = test_data.drop(columns=['ID'])  # Assuming your test dataset is in a 'test' dataframe
y_test=test_data['ID']

print("Number of features in training data:", X_train.shape[1])
print("Number of features in test data:", X_test.shape[1])

# Initialize the LabelEncoder to encode labels
label_encoder = LabelEncoder()

# Fit and transform labels in the training data
y_train = label_encoder.fit_transform(y_train)

# Standardize the features in both the training and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Feedforward Neural Network (FNN) model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)

#printing accuracy:
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Inverse transform the encoded predictions to original labels
predicted_labels = label_encoder.inverse_transform(y_pred)

#save the predicted labels in a csv file in the format 'ID', 'Predicted_Label'
submission = pd.DataFrame({'ID': y_test, 'Predicted_Label': predicted_labels})

# Save the submission file
submission.to_csv('C://Users//91991//Desktop//MLBA//Assignment2//submission-cnn.csv', index=False)