from data_processing import load_data, preprocess_data, split_data
from classification_model import train_model, evaluate_model

# Load and preprocess the data
X, y = load_data("data/ds_salaries.csv")
X_encoded = preprocess_data(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = split_data(X_encoded, y)

# Train the classifier
clf = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(clf, X_test, y_test)
