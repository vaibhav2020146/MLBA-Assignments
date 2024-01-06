import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load your training dataset
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//train.csv')

# Drop the 'ID' column from the training data
#train_data = train_data.drop(columns=['ID'])

# Split the training data into features (X) and labels (y)
X = train_data.drop(columns=['Labels'])
y = train_data['Labels']

# Standardize the training data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and train a K-NN classifier (you may choose the best k based on cross-validation)
best_k = 8  # You can choose the best k based on cross-validation
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X, y)

# Load your test dataset
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//test.csv')

# Drop the 'ID' column from the test data
X_test= test_data.drop(columns=['ID'])
y_test=test_data['ID']

# Standardize the test data using the same scaler
X_test = scaler.transform(X_test)

# Use the trained K-NN model to make predictions on the test data
test_predictions = best_knn.predict(X_test)

# Create a DataFrame to store the test predictions
test_results = pd.DataFrame({'ID': y_test, 'Labels': test_predictions})

# Save the predictions in a CSV file in the format of sample.csv
test_results.to_csv('C://Users//91991//Desktop//MLBA//Assignment2//submission-knn.csv', index=False)
