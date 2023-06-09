"""
This is a boilerplate pipeline
generated using Kedro 0.18.10
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import wandb

wandb.init(
    project="asi-final",
)   

def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    data_train = data.sample(
        frac=parameters["train_fraction"], random_state=parameters["random_state"]
    )
    data_test = data.drop(data_train.index)

    X_train = data_train.drop(columns=parameters["target_column"])
    X_test = data_test.drop(columns=parameters["target_column"])
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test


import wandb
from wandb.data_types import Table

def make_predictions(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> pd.Series:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """
    wandb.log({"message": "Starting make_predictions function"})  # Log a message indicating the start of the function

    categorical_columns = ['experience_level', 'employment_type']
    numeric_columns = ['work_year']

    # Perform one-hot encoding for categorical columns
    X_train_encoded = pd.get_dummies(X_train[categorical_columns])
    X_test_encoded = pd.get_dummies(X_test[categorical_columns])

    # Concatenate encoded categorical columns and numeric columns
    X_train_combined = pd.concat([X_train_encoded, X_train[numeric_columns]], axis=1)
    X_test_combined = pd.concat([X_test_encoded, X_test[numeric_columns]], axis=1)

    X_train_numpy = X_train_combined.astype(float).to_numpy()
    X_test_numpy = X_test_combined.astype(float).to_numpy()

    squared_distances = np.sum(
        (X_train_numpy[:, None] - X_test_numpy[None, :]) ** 2, axis=-1
    )
    nearest_neighbour = squared_distances.argmin(axis=0)
    y_pred = pd.Series(y_train.iloc[nearest_neighbour].values, index=X_test.index, name='salary')

    # Convert prediction values to a DataFrame
    prediction_df = pd.DataFrame({"predictions": y_pred.values}, index=X_test.index)

    # Log the predictions as a table in WandB
    wandb.log({"predictions_table": Table(dataframe=prediction_df)})  

    return y_pred




def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    accuracy = (y_pred == y_test).sum() / len(y_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
