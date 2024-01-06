import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Load your training dataset (replace 'train.csv' with your actual file)
train_data = h2o.import_file('C://Users//91991//Desktop//MLBA//train.csv')

# Load your test dataset (replace 'test.csv' with your actual file)
test_data = h2o.import_file('C://Users//91991//Desktop//MLBA//test.csv')

# Define predictor columns (features) and target column
predictors = train_data.columns[:-1]  # Assuming the last column is the target
target = 'Label'

# Split data into training and validation sets
train, valid = train_data.split_frame(ratios=[0.8])

# Initialize and train AutoML
aml = H2OAutoML(max_models=20, seed=1)  # Adjust max_models as needed
aml.train(x=predictors, y=target, training_frame=train, validation_frame=valid)

# Get the best model from AutoML
best_model = aml.leader

# Make predictions on the test dataset
predictions = best_model.predict(test_data)

# Save predictions to a CSV file
predictions_df = predictions.as_data_frame()
predictions_df.to_csv('C://Users//91991//Desktop//MLBA//predictions-h2o.csv', index=False)

# Shutdown H2O
h2o.shutdown()
