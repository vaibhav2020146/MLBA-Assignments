'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import PowerTransformer

# Load your training dataset
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//train.csv')

# Split the training data into features (X) and labels (y)
X = train_data.drop(columns=['Labels'])
y = train_data['Labels']

# Feature selection: Select the top k features based on ANOVA F-statistics
k = 200  # Adjust k based on your data
selector = SelectKBest(score_func=f_classif, k=k)
X = selector.fit_transform(X, y)

# Standardize the training data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a deep neural network model
model = Sequential()
model.add(Dense(units=128, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Implement early stopping
#from keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Load your test dataset
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//test.csv')

# Feature selection for the test data
X_test = selector.transform(test_data.drop(columns=['ID']))

# Standardize the test data using the same scaler
X_test = scaler.transform(X_test)

# Predict the labels for the test data
test_predictions = model.predict(X_test)
test_predictions = (test_predictions > 0.5).astype(int)

# Create a DataFrame to store the test predictions
test_results = pd.DataFrame({'ID': test_data['ID'], 'Labels': test_predictions.flatten()})

# Save the predictions in a CSV file in the format of sample.csv
test_results.to_csv('C://Users//91991//Desktop//MLBA//Assignment2//submission-dnn.csv', index=False)'''



'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load your training dataset
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//train.csv')

# Split the training data into features (X) and labels (y)
X = train_data.drop(columns=['Labels'])
y = train_data['Labels']

# Apply feature selection (SelectKBest) and standardization
k = 100  # Adjust k based on your data
selector = SelectKBest(score_func=f_classif, k=k)
X = selector.fit_transform(X, y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

# Apply t-SNE for further dimensionality reduction on both training and validation data
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # Adjust parameters
X_train_tsne = tsne.fit_transform(X_train_pca)
X_val_tsne = tsne.fit_transform(X_val_pca)

# Create a deep neural network model
model = Sequential()
model.add(Dense(units=128, input_dim=X_train_tsne.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train_tsne, y_train, epochs=100, batch_size=32, validation_data=(X_val_tsne, y_val))

# Load your test dataset
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//test.csv')

# Apply feature selection, standardization, PCA, and t-SNE for the test data
X_test = selector.transform(test_data.drop(columns=['ID']))
X_test = scaler.transform(X_test)
X_test_pca = pca.transform(X_test)

# Apply t-SNE to the test data
X_test_tsne = tsne.fit_transform(X_test_pca)

# Predict the labels for the test data
test_predictions = model.predict(X_test_tsne)
test_predictions = (test_predictions > 0.5).astype(int)

# Create a DataFrame to store the test predictions
test_results = pd.DataFrame({'ID': test_data['ID'], 'Labels': test_predictions.flatten()})

# Save the predictions in a CSV file in the format of sample.csv
test_results.to_csv('C://Users//91991//Desktop//MLBA//Assignment2//submission-dnn.csv', index=False)'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import PowerTransformer

# Load your training dataset
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//train.csv')

# Split the training data into features (X) and labels (y)
X = train_data.drop(columns=['Labels'])
y = train_data['Labels']

# Feature selection: Select the top k features based on ANOVA F-statistics
k = 200  # Adjust k based on your data
selector = SelectKBest(score_func=f_classif, k=k)
X = selector.fit_transform(X, y)

# Standardize the training data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a deep neural network model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Implement early stopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Load your test dataset
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//Assignment2//test.csv')

# Feature selection for the test data
X_test = selector.transform(test_data.drop(columns=['ID']))

# Standardize the test data using the same scaler
X_test = scaler.transform(X_test)

# Predict the labels for the test data
test_predictions = model.predict(X_test)
test_predictions = (test_predictions > 0.5).astype(int)

# Create a DataFrame to store the test predictions
test_results = pd.DataFrame({'ID': test_data['ID'], 'Labels': test_predictions.flatten()})

# Save the predictions in a CSV file in the format of sample.csv
test_results.to_csv('C://Users//91991//Desktop//MLBA//Assignment2//submission-dnn.csv', index=False)