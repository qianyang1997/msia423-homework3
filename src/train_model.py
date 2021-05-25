import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier

import logging

logger2 = logging.getLogger(__name__)


def train_test_split(features: pd.DataFrame, target: pd.DataFrame, train_feature_path: str,
                     train_target_path: str, test_feature_path: str, test_target_path: str,
                     seed: int, test_size: int) -> tuple:
    """split train test sets.

    :param features: :obj: pandas dataframe - csv of features
    :param target: :obj: pandas dataframe - csv of labels
    :param train_feature_path: str - file path to save training features
    :param train_target_path: str - file path to save training labels
    :param test_feature_path: str - file path to save test features
    :param test_target_path: str - file path to save test labels
    :param seed: int - random state for train test split
    :param test_size: float or int - if float, it needs to be between 0-1 and represents proportion
                                     of data in test set; if int, it represents number of data points
                                     in test set
    :return:
        :obj: tuple - tuple of pandas dataframes consisting of features and labels divided into train
                      and test sets
    """
    X_train, X_test, y_train, y_test = tts(features, target, random_state=seed, test_size=test_size)

    X_train.to_csv(train_feature_path, index=False)
    X_test.to_csv(train_target_path, index=False)
    y_train.to_csv(test_feature_path, index=False)
    y_test.to_csv(test_target_path, index=False)
    logger2.info('Features and labels written to file.')

    return X_train, X_test, y_train, y_test


def fit_model(X_train: pd.DataFrame, labels: pd.DataFrame, X_test: pd.DataFrame,
              features_list: list, output_path: str, **kwargs) -> pd.DataFrame:
    """fit a random forest model based on custom sklearn parameters and make prediction on test set.

    :param X_train: :obj: pandas dataframe - training features
    :param labels:  :obj: pandas dataframe - training labels
    :param X_test: :obj: pandas dataframe - test features
    :param features_list: list - list of strings, features to use in the model
    :param output_path: str - output path to save predictions
    :param kwargs: dict - parameters compatible with sklearn RandomForestClassifier organized in a dictionary. See
                          https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                          for details.
    :return: :obj: pandas dataframe - dataframe consisting of predictions on test set
    """
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X_train[features_list], labels)

    ypred_proba = rf.predict_proba(X_test[features_list])[:, 1]
    ypred_bin = rf.predict(X_test[features_list])

    df = pd.DataFrame(ypred_proba, columns=['ypred_proba'])
    df['ypred_bin'] = ypred_bin
    df.to_csv(output_path, index=False)
    logger2.info(f'Predictions on test set saved to {output_path}.')

    return df
