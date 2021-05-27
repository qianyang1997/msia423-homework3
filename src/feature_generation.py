import pandas as pd
import numpy as np


def feature_gen(data: pd.DataFrame) -> tuple:
    """generate features and labels and write to local csv.

    :param data: :obj: pandas dataframe - data to be fed
    :returns tuple - tuple of pandas dataframe consisting of features and labels
    """
    features = data.drop('class', axis=1)
    target = data['class']

    return features, target


def add_log_entropy(features: pd.DataFrame) -> pd.DataFrame:
    """Add log entropy as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :return: :obj pandas dataframe - output features with additional column
    """
    features['log_entropy'] = features.visible_entropy.apply(np.log)

    return features


def add_entropy_x_contrast(features: pd.DataFrame) -> pd.DataFrame:
    """Add entropy x contrast as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :return: :obj pandas dataframe - output features with additional column
    """
    features['entropy_x_contrast'] = features.visible_contrast.multiply(
        features.visible_entropy)

    return features


def add_ir_range(features: pd.DataFrame) -> pd.DataFrame:
    """Add IR range as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :return: :obj pandas dataframe - output features with additional column
    """
    features['IR_range'] = features.IR_max - features.IR_min

    return features


def add_ir_norm_range(features: pd.DataFrame) -> pd.DataFrame:
    """Add IR norm range as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :return: :obj pandas dataframe - output features with additional column
    """
    features['IR_norm_range'] = (features.IR_max - features.IR_min).divide(
        features.IR_mean)

    return features
