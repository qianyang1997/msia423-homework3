import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier

import logging

logger2 = logging.getLogger(__name__)


def train_test_split(features: pd.DataFrame, target: pd.DataFrame, seed: int, test_size: int) -> tuple:
    """split train test sets.

    :param features: :obj: pandas dataframe - csv of features
    :param target: :obj: pandas dataframe - csv of labels
    :param seed: int - random state for train test split
    :param test_size: float or int - if float, it needs to be between 0-1 and represents proportion
                                     of data in test set; if int, it represents number of data points
                                     in test set
    :return:
        :obj: tuple - tuple of pandas dataframes consisting of features and labels divided into train
                      and test sets
    """
    X_train, X_test, y_train, y_test = tts(features, target, random_state=seed, test_size=test_size)

    return X_train, X_test, y_train, y_test


def fit_model(X_train: pd.DataFrame, labels: pd.DataFrame, X_test: pd.DataFrame,
              features_list: list, **kwargs) -> pd.DataFrame:
    """fit a random forest model based on custom sklearn parameters and make prediction on test set.

    :param X_train: :obj: pandas dataframe - training features
    :param labels:  :obj: pandas dataframe - training labels
    :param X_test: :obj: pandas dataframe - test features
    :param features_list: list - list of strings, features to use in the model
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
    logger2.info(f'Predictions df created.')

    return df
