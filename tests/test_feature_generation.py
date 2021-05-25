import pytest
import yaml

import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_generation import feature_gen, add_log_entropy, \
    add_ir_norm_range, add_ir_range, add_entropy_x_contrast, logger1

logger1.setLevel('ERROR')

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    feature_config = config['generate_features']
    data_config = config['load_data']


def test_feature_gen_happy():

    features_true = pd.read_csv(feature_config['feature_gen']['features_output_path'])

    data = pd.read_csv(data_config['output_path'])
    features_test, labels_test = feature_gen(data)

    assert_frame_equal(features_true, features_test)


def test_feature_gen_unhappy():

    with pytest.raises(FileNotFoundError):
        features_true = pd.read_csv('data/raw_features.csv')


def test_add_log_entropy_happy():

    features_true = pd.read_csv(feature_config['add_log_entropy']['output_path'])

    data = pd.read_csv(feature_config['feature_gen']['features_output_path'])
    features_test = add_log_entropy(data)

    assert_frame_equal(features_true, features_test)


def test_add_log_entropy_unhappy():

    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_log_entropy(data)


def test_add_entropy_x_contrast_happy():

    features_true = pd.read_csv(feature_config['add_entropy_x_contrast']['output_path'])

    data = pd.read_csv(feature_config['add_log_entropy']['output_path'])
    features_test = add_entropy_x_contrast(data)

    assert_frame_equal(features_true, features_test)


def test_add_entropy_x_contrast_unhappy():

    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_entropy_x_contrast(data)


def test_add_ir_range_happy():

    features_true = pd.read_csv(feature_config['add_ir_range']['output_path'])

    data = pd.read_csv(feature_config['add_entropy_x_contrast']['output_path'])
    features_test = add_ir_range(data)

    assert_frame_equal(features_true, features_test)


def test_add_ir_range_unhappy():

    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_ir_range(data)


def test_add_ir_norm_range_happy():

    features_true = pd.read_csv(feature_config['add_ir_norm_range']['output_path'])

    data = pd.read_csv(feature_config['add_ir_range']['output_path'])
    features_test = add_ir_norm_range(data)

    assert_frame_equal(features_true, features_test)


def test_add_ir_norm_range_unhappy():

    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_ir_norm_range(data)
