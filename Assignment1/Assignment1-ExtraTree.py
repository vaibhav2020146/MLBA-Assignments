import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load your training dataset
df = pd.read_csv('C://Users//91991//Desktop//MLBA//train.csv')

# Load your test dataset
df_test = pd.read_csv('C://Users//91991//Desktop//MLBA//test.csv')

# Extract features from protein sequences using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(df['Sequence'])
X_test = vectorizer.transform(df_test['Sequence'])

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, df['Label'], test_size=0.2, random_state=42)

# Train the Extra Trees Classifier
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = clf.predict(X_val)

#prin the prediction:
print(y_pred)

# Calculate and print the validation accuracy
validation_accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {validation_accuracy:.4f}')

# Predict on the test set
y_test_pred = clf.predict(X_test)

# Save the predictions to a CSV file with ID
df_test['Label'] = y_test_pred
df_test[['ID', 'Label']].to_csv('C://Users//91991//Desktop//MLBA//output-extra.csv', index=False)

#now compare the accuracy with respect to each ID from output.csv and sample.csv:
df3 = pd.read_csv('C://Users//91991//Desktop//MLBA//sample.csv')
df4 = pd.read_csv('C://Users//91991//Desktop//MLBA//output-extra.csv')
df5 = pd.merge(df3, df4, on='ID')

#now check the accuracy of the model:
from sklearn.metrics import accuracy_score
accuracy_score(df5['Label_x'], df5['Label_y'])
#print the accuracy
print(accuracy_score(df5['Label_x'], df5['Label_y']))