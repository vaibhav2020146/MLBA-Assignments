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
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))
X = vectorizer.fit_transform(df['Sequence'])
X1 = vectorizer.transform(df1['Sequence'])

#train the model for int based labels:
Y = df['Label']
#Y1 = df1['ID']

#train the model
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X, Y)

#predict the model
Z = logistic.predict(X1)
print(Z)
#write the output in new csv file
#df1['Label'] = Z
#df1.to_csv('C://Users//91991//Desktop//MLBA//output.csv', index=False)

#now save the output Z in new csv file along with ID column from df1
df2 = pd.DataFrame({'ID':df1['ID'], 'Label':Z})
df2.to_csv('C://Users//91991//Desktop//MLBA//output.csv', index=False)

#now check the accuracy of the model:
'''from sklearn.metrics import accuracy_score
accuracy_score(df['Label'], logistic.predict(X))
#print the accuracy
print(accuracy_score(df['Label'], logistic.predict(X)))'''

#now compare the accuracy with respect to each ID from output.csv and sample.csv:
df3 = pd.read_csv('C://Users//91991//Desktop//MLBA//sample.csv')
df4 = pd.read_csv('C://Users//91991//Desktop//MLBA//output.csv')
df5 = pd.merge(df3, df4, on='ID')
print(df5)
#now check the accuracy of the model:
from sklearn.metrics import accuracy_score
accuracy_score(df5['Label_x'], df5['Label_y'])
#print the accuracy
print(accuracy_score(df5['Label_x'], df5['Label_y']))