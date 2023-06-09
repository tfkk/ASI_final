import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv('data/ds_salaries.csv')
    X = df.drop('salary', axis=1)
    y = df['salary']
    return X, y

def preprocess_data(X):
    X_encoded = pd.get_dummies(X)
    return X_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
