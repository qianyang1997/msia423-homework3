import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import logging

logger1 = logging.getLogger(__name__)


def feature_gen(data: pd.DataFrame, train_feature_path: str, train_target_path: str,
                test_feature_path: str, test_target_path: str, seed: int, test_size: int) -> tuple:
    """generate features and labels and write to local csv.

    :param data: :obj: pandas dataframe - data to be fed
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
    features = data.drop('class', axis=1)                                      
    target = data['class']

    return features, target


def add_log_entropy(features: pd.DataFrame) -> pd.DataFrame:

    features['log_entropy'] = features.visible_entropy.apply(np.log)

    return features


def add_entropy_x_contrast(features: pd.DataFrame) -> pd.DataFrame:

    features['entropy_x_contrast'] = features.visible_contrast.multiply(
        features.visible_entropy)

    return features


def add_IR_range(features: pd.DataFrame) -> pd.DataFrame:

    features['IR_range'] = features.IR_max - features.IR_min

    return features


def add_IR_norm_range(features: pd.DataFrame) -> pd.DataFrame:

    features['IR_norm_range'] = (features.IR_max - features.IR_min).divide(
        features.IR_mean)

    return features


def train_test_split(features: pd.DataFrame, target: pd.DataFrame, train_feature_path: str,
                     train_target_path: str, test_feature_path: str, test_target_path: str,
                     seed: int, test_size: int) -> tuple:
    """generate features and labels and write to local csv.

    :param features:
    :param target:
    :param train_feature_path:
    :param train_target_path:
    :param test_feature_path:
    :param test_target_path:
    :param seed:
    :param test_size:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=seed,
                                                        test_size=test_size)

    X_train.to_csv(train_feature_path, index=False)
    X_test.to_csv(train_target_path, index=False)
    y_train.to_csv(test_feature_path, index=False)
    y_test.to_csv(test_target_path, index=False)
    logger1.info('Features and labels written to file.')

    return X_train, X_test, y_train, y_test


