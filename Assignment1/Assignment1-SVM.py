import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.io import imread,imshow
import sklearn
from sklearn import linear_model
import pandas as pd


#read csv file
df = pd.read_csv('C://Users//91991//Desktop//MLBA//train.csv')

#read test dataset
df1 = pd.read_csv('C://Users//91991//Desktop//MLBA//test.csv')

#train the model for string based sequences:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))  # Experiment with different ngram ranges
X = vectorizer.fit_transform(df['Sequence'])
X1 = vectorizer.transform(df1['Sequence'])

#train the model for int based labels:
Y = df['Label']

#split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#train the model using SVM classifier so as to achieve maximum accuracy:
from sklearn import svm
clf = svm.SVC(kernel='poly', degree=2, C=0.1, gamma='scale', coef0=0.1, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
clf.fit(X_train, y_train)

#predict the model
Z = clf.predict(X1)
print(Z)

#now save the output Z in new csv file along with ID column from df1
df2 = pd.DataFrame({'ID':df1['ID'], 'Label':Z})
df2.to_csv('C://Users//91991//Desktop//MLBA//output-svm.csv', index=False)

#now compare the accuracy with respect to each ID from output.csv and sample.csv:
df3 = pd.read_csv('C://Users//91991//Desktop//MLBA//sample.csv')
df4 = pd.read_csv('C://Users//91991//Desktop//MLBA//output-svm.csv')
df5 = pd.merge(df3, df4, on='ID')

#now check the accuracy of the model:
from sklearn.metrics import accuracy_score
accuracy_score(df5['Label_x'], df5['Label_y'])
#print the accuracy
print(accuracy_score(df5['Label_x'], df5['Label_y']))