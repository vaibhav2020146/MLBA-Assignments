import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load and preprocess the training data
df_train = pd.read_csv('C://Users//91991//Desktop//MLBA//train.csv')
tokenizer = Tokenizer(num_words=1000, char_level=True)
tokenizer.fit_on_texts(df_train['Sequence'])
X_train_seq = tokenizer.texts_to_sequences(df_train['Sequence'])
X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding="post", truncating="post")
y_train = df_train['Label']

# Build and compile the CNN model
model = Sequential()
model.add(Embedding(1000, 128, input_length=X_train_pad.shape[1]))
model.add(Conv1D(256, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=20, batch_size=256)

# Save the trained model to a file
model.save('C://Users//91991//Desktop//MLBA//protein_classification_cnn_model.h5')