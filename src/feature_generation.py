import pandas as pd
import numpy as np

import logging

logger1 = logging.getLogger(__name__)


def feature_gen(data: pd.DataFrame, features_output_path: str, labels_output_path: str) -> tuple:
    """generate features and labels and write to local csv.

    :param data: :obj: pandas dataframe - data to be fed
    :param features_output_path: str - output path for features
    :param labels_output_path: str -  output path for labels
    :returns tuple - tuple of pandas dataframe consisting of features and labels
    """
    features = data.drop('class', axis=1)                                      
    target = data['class']

    features.to_csv(features_output_path, index=False)
    target.to_csv(labels_output_path, index=False)
    logger1.info('Features and labels written to files.')

    return features, target


def add_log_entropy(features: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Add log entropy as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :param output_path: str - output path for new features csv
    :return: :obj pandas dataframe - output features with additional column
    """
    features['log_entropy'] = features.visible_entropy.apply(np.log)

    features.to_csv(output_path, index=False)
    logger1.info(f'New feature dataframe written to {output_path}.')

    return features


def add_entropy_x_contrast(features: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Add entropy x contrast as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :param output_path: str - output path for new features csv
    :return: :obj pandas dataframe - output features with additional column
    """
    features['entropy_x_contrast'] = features.visible_contrast.multiply(
        features.visible_entropy)

    features.to_csv(output_path, index=False)
    logger1.info(f'New feature dataframe written to {output_path}.')

    return features


def add_ir_range(features: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Add IR range as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :param output_path: str - output path for new features csv
    :return: :obj pandas dataframe - output features with additional column
    """
    features['IR_range'] = features.IR_max - features.IR_min

    features.to_csv(output_path, index=False)
    logger1.info(f'New feature dataframe written to {output_path}.')

    return features


def add_ir_norm_range(features: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Add IR norm range as a feature.

    :param features: :obj: pandas dataframe - features to be fed
    :param output_path: str - output path for new features csv
    :return: :obj pandas dataframe - output features with additional column
    """
    features['IR_norm_range'] = (features.IR_max - features.IR_min).divide(
        features.IR_mean)

    features.to_csv(output_path, index=False)
    logger1.info(f'New feature dataframe written to {output_path}.')

    return features
