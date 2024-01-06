import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.io import imread,imshow
import sklearn
from sklearn import linear_model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#read csv file
df = pd.read_csv('C://Users//91991//Desktop//MLBA//train.csv')

#read test dataset
df1 = pd.read_csv('C://Users//91991//Desktop//MLBA//test.csv')

#train the model for string based sequences for CNN using 2D convolution:
tokenizer = Tokenizer(num_words=1000, char_level=True)
tokenizer.fit_on_texts(df['Sequence'])
X = tokenizer.texts_to_sequences(df['Sequence'])
X1 = tokenizer.texts_to_sequences(df1['Sequence'])

#pad the sequences
X = pad_sequences(X, maxlen=100,padding="post", truncating="post")
X1 = pad_sequences(X1, maxlen=100,padding="post", truncating="post")

#train the model for int based labels:
Y = df['Label']

#train the model using 2d CNN:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()
model.add(Embedding(1000, 128, input_length=X.shape[1]))
#add 2d convolution layer
model.add(Conv2D(64, kernel_size=(3, 3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#add more layers to improve the accuracy
model.add(Conv2D(64, kernel_size=(3, 3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=64)

#predict the model
Z = model.predict(X1)
print(Z)

#now convert the float values to int to get the output in 0 and 1
Z = np.round(Z)
print(Z)
Z=Z.flatten()
#now save the output Z in new csv file along with ID column from df1
df2 = pd.DataFrame({'ID':df1['ID'], 'Label':Z})
df2.to_csv('C://Users//91991//Desktop//MLBA//output-resnet.csv', index=False)

#now compare the accuracy with respect to each ID from output.csv and sample.csv:
df3 = pd.read_csv('C://Users//91991//Desktop//MLBA//sample.csv')
df4 = pd.read_csv('C://Users//91991//Desktop//MLBA//output-resnet.csv')
df5 = pd.merge(df3, df4, on='ID')
print(df5)
#now check the accuracy of the model:
from sklearn.metrics import accuracy_score
accuracy_score(df5['Label_x'], df5['Label_y'])
#print the accuracy
print(accuracy_score(df5['Label_x'], df5['Label_y']))