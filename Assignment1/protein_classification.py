#---------------------------------------------------------------------GROUP-25---------------------------------------------------------------------#
from tensorflow.keras.preprocessing.text import Tokenizer
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

def load_data(train_file, test_file):
    # Load the dataset
    data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return data, test_data

def make_predictions(model, X_test, label_encoder):
    # Make predictions on the test data
    predictions = model.predict(X_test)#predicts the labels for the test data
    dictionary = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6}
    predicted_labels = (predictions > 0.5).astype(int)#converts the predictions into 0 or 1 based on the threshold of 0.5

    # Convert back to original labels
    predicted_labels = label_encoder.inverse_transform(predicted_labels.flatten())#converts the labels back to the original labels
    return predicted_labels#returns the predicted labels

def tokenize_data(data):
    # Tokenize the data
    tokenizer = Tokenizer(char_level=True)#Tokenizer is used to tokenize the data. char_level=True tells the tokenizer to tokenize the data at the character level.
    check=True
    tokenizer.fit_on_texts(data['Sequence'])#fit_on_texts is used to fit the tokenizer on the data. It creates a vocabulary of all the characters in the data.
    return tokenizer

def pad_sequence(data, tokenizer):
    # Pad the sequence
    X = tokenizer.texts_to_sequences(data['Sequence'])#converts each sequence into a list of integers. Each integer represents the index of the character in the vocabulary.
    l=[len(i) for i in X]
    X = pad_sequences(X)#pad_sequences is used to ensure that all sequences in a list have the same length. By default this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence.
    return X

def label_encode(data):
    # Encode labels
    label_encoder = LabelEncoder()#LabelEncoder is used to encode the labels into integers.
    y = label_encoder.fit_transform(data['Label'])#fit_transform is used to fit label encoder and return encoded labels
    return y

def preprocess_data(data, test_data):#we are preprocessing data because the data is not in the format that the model can understand. So we are converting the data into a format that the model can understand.
    tokenizer = tokenize_data(data)#tokenizes the data
    X = pad_sequence(data, tokenizer)#pads the sequence
    coding=(len(tokenizer.word_index) + 1)
    y = label_encode(data)#encodes the labels
    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)#split the data into training and validation sets
    return tokenizer, X_train, X_valid, y_train, y_valid, test_data

def Embadding_model(model,tokenizer, max_sequence_length):
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_sequence_length))#Embedding layer is used to create word embeddings. It takes the integer encoded data as input, and the output is the word embeddings for each word in the sequence.

def LSTM_model(model):
    model.add(Bidirectional(LSTM(64, return_sequences=True)))#Bidirectional layer is used to make the model learn from the input sequence both forward and backward directions.
    model.add(Bidirectional(LSTM(32)))#Bidirectional layer is used to make the model learn from the input sequence both forward and backward directions.
    
def add_dense_layer(model):
    model.add(Dense(64, activation='relu'))#Dense layer is used to create a fully connected layer.
    model.add(Dropout(0.5))#Dropout layer is used to prevent overfitting and 0.5 tells the model to drop 50% of the neurons. so that the model can learn from the remaining 50%.
    model.add(Dense(1, activation='sigmoid'))#Dense layer is used to create a fully connected layer, used so that we can easily do binary classification.

def build_model(tokenizer, max_sequence_length):
    # Build the model and for activation function we have used relu and sigmoid because relu is used for the hidden layers and sigmoid is used for the output layer.
    model = Sequential()#Sequential groups a linear stack of layers into a tf.keras.Model.
    Embadding_model(model,tokenizer, max_sequence_length)
    LSTM_model(model)
    add_dense_layer(model)
    return model

def train_model(model, X_train, y_train, X_valid, y_valid):
    # Compile the model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])#Adam optimizer is used to optimize the model. Binary crossentropy is used as the loss function.
    epochs_number_to_train = 20
    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_valid, y_valid))#Train the model for 15 epochs with a batch size of 64.

def preprocess_test_data(tokenizer, test_data, max_sequence_length):
    # Preprocess the test data
    X_test = tokenizer.texts_to_sequences(test_data['Sequence'])#converts each sequence into a list of integers. Each integer represents the index of the character in the vocabulary.
    check_if_sequence_is_padded=True
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)#pad_sequences is used to ensure that all sequences in a list have the same length. By default this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence.
    return X_test

def save_submission_file(test_data, predicted_labels, submission_file):
    # Create a DataFrame for submission
    submission_df = pd.DataFrame({'ID': test_data['ID'], 'Label': predicted_labels})#creates a dataframe with the ID and predicted labels
    submission_made=True
    submission_not_made=False
    # Save the submission file
    submission_df.to_csv(submission_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Protein Classification Model Trainer")#creates a parser object
    parser.add_argument("--train_file", required=True, help="C://Users//91991//Desktop//MLBA")#adds an argument to the parser object
    parser.add_argument("--test_file", required=True, help="C://Users//91991//Desktop//MLBA")
    parser.add_argument("--submission_file", required=True, help="C://Users//91991//Desktop//MLBA")

    args = parser.parse_args()

    data, test_data = load_data(args.train_file, args.test_file)#loads the data
    tokenizer, X_train, X_valid, y_train, y_valid, test_data = preprocess_data(data, test_data)#preprocesses the data
    model = build_model(tokenizer, X_train.shape[1])#builds the model
    train_model(model, X_train, y_train, X_valid, y_valid)#trains the model
    X_test = preprocess_test_data(tokenizer, test_data, X_train.shape[1])#preprocesses the test data
    label_encoder = LabelEncoder()#LabelEncoder is used to encode the labels into integers.
    label_encoder.fit(data['Label'])#fit is used to fit label encoder and return encoded labels
    predicted_labels = make_predictions(model, X_test, label_encoder)#makes predictions on the test data
    save_submission_file(test_data, predicted_labels, args.submission_file)#saves the submission file

if __name__ == "__main__":
    main()
