# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  # You can replace with the model of your choice

# Load the training and test datasets
train_data = pd.read_csv("C://Users//91991//Desktop//MLBA//Assignment2//train.csv")
test_data = pd.read_csv("C://Users//91991//Desktop//MLBA//Assignment2//test.csv")

# Separate the features (gene expression) and labels in the training data
X = train_data.iloc[:, 1:]  # Features
y = train_data.iloc[:, 0]   # Labels

# Data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model's performance on the validation set
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Train the model on the entire training data
model.fit(X, y)

# Data preprocessing for the test data
X_test = test_data.iloc[:, 1:]  # Features
X_test = scaler.transform(X_test)

# Make predictions on the test data
test_predictions = model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({'ID': test_data.iloc[:, 0], 'Predicted_Label': test_predictions})
submission.to_csv("C://Users//91991//Desktop//MLBA//Assignment2//submission.csv", index=False)
