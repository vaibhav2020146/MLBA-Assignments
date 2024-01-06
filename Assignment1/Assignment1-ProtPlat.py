from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load training and testing data from CSV files
train_data = pd.read_csv('C://Users//91991//Desktop//MLBA//train.csv')
test_data = pd.read_csv('C://Users//91991//Desktop//MLBA//test.csv')

# Extract protein sequences and labels from the training data
train_sequences = train_data['Sequence']
train_labels = train_data['Label']

# Tokenize the training sequences
train_encodings = tokenizer(train_sequences.tolist(), padding=True, truncation=True, return_tensors="pt")

# Create a tensor of labels with the same shape as logits
train_labels = torch.tensor(train_labels.tolist())
train_labels = train_labels.view(-1, 1)  # Reshape to match the shape of logits

batch_size = 4


train_inputs = tokenizer(train_sequences.tolist(), padding=True, truncation=True, return_tensors="pt")
train_labels = torch.tensor(train_labels.tolist())
test_inputs = tokenizer(test_sequences.tolist(), padding=True, truncation=True, return_tensors="pt")

train_dataset = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_labels)# to import it write: from torch.utils.data import TensorDataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#to import it write: from torch.utils.data import DataLoader

# Use mixed-precision training
model = model.half()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop with gradient accumulation
num_accumulation_steps = 4  # Accumulate gradients over 4 mini-batches
for epoch in range(3):
    optimizer.zero_grad()
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids, attention_mask, labels = input_ids.half(), attention_mask.half(), labels
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / num_accumulation_steps
        loss.backward()
        if (batch_idx + 1) % num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()



# Evaluate the model on the testing data
test_sequences = test_data['Sequence']
test_encodings = tokenizer(test_sequences.tolist(), padding=True, truncation=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    logits = model(**test_encodings).logits
    predictions = torch.argmax(logits, dim=1).tolist()

# Save the predictions to a CSV file
test_data['Predicted_Label'] = predictions
test_data.to_csv('C://Users//91991//Desktop//MLBA//test_predictions.csv', index=False)

#now compare the accuracy with respect to each ID from output.csv and sample.csv:
df3 = pd.read_csv('C://Users//91991//Desktop//MLBA//sample.csv')
df4 = pd.read_csv('C://Users//91991//Desktop//MLBA//test_predictions.csv')
df5 = pd.merge(df3, df4, on='ID')

#now check the accuracy of the model:
from sklearn.metrics import accuracy_score
accuracy_score(df5['Label_x'], df5['Predicted_Label'])
#print the accuracy
print(accuracy_score(df5['Label_x'], df5['Predicted_Label']))
