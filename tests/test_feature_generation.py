import pytest
import yaml

import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_generation import feature_gen, add_log_entropy, \
    add_ir_norm_range, add_ir_range, add_entropy_x_contrast

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    feature_config = config['generate_features']
    data_config = config['load_data']


def test_feature_gen_happy():
    data = pd.DataFrame(
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'class'],
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, 0.0],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.834, 167.0, 239.0, 213.7188, 0.0],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, 0.0],
            [4.0, 197.0, 77.4805, 0.089, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773, 0.0],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, 0.0]
        ]
    )

    features_true = pd.DataFrame(
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min'],
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195],
        ]
    )

    features_test, labels_test = feature_gen(data)
    assert_frame_equal(features_true, features_test)


def test_feature_gen_unhappy():
    data = pd.DataFrame(
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min'],
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555]
        ]
    )

    with pytest.raises(KeyError):
        feature_gen(data)


def test_add_log_entropy_happy():
    data = pd.DataFrame(
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min'],
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195],
        ]
    )

    features_true = pd.DataFrame(
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy'],
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064]
        ]
    )

    features_test = add_log_entropy(data)

    assert_frame_equal(features_true, features_test)


def test_add_log_entropy_unhappy():
    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_log_entropy(data)


def test_add_entropy_x_contrast_happy():
    data = pd.DataFrame(
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy'],
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064]
        ]
    )

    features_true = pd.DataFrame(
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646,
             21.916179179999997],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645, 17.87952369],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632,
             21.09170772],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345, 21.24964287],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064,
             18.065510980000003]
        ],
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy', 'entropy_x_contrast']
    )

    features_test = add_entropy_x_contrast(data)
    assert_frame_equal(features_true, features_test)


def test_add_entropy_x_contrast_unhappy():
    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_entropy_x_contrast(data)


def test_add_ir_range_happy():
    data = pd.DataFrame(
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646,
             21.916179179999997],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645, 17.87952369],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632,
             21.09170772],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345, 21.24964287],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064,
             18.065510980000003]
        ],
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy', 'entropy_x_contrast']
    )

    features_true = pd.DataFrame(
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646,
             21.916179179999997, 26.644499999999994],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645, 17.87952369, 25.281200000000013],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632,
             21.09170772, 12.41409999999999],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345, 21.24964287, 41.7227],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064,
             18.065510980000003, 49.980500000000006]
        ],
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy', 'entropy_x_contrast',
                 'IR_range']
    )
    features_test = add_ir_range(data)
    assert_frame_equal(features_true, features_test)


def test_add_ir_range_unhappy():
    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_ir_range(data)


def test_add_ir_norm_range_happy():

    data = pd.DataFrame(
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646,
             21.916179179999997, 26.644499999999994],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645, 17.87952369, 25.281200000000013],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632,
             21.09170772, 12.41409999999999],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345, 21.24964287, 41.7227],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064,
             18.065510980000003, 49.980500000000006]
        ],
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy', 'entropy_x_contrast',
                 'IR_range']
    )
    features_true = pd.DataFrame(
        data=[
            [3.0, 140.0, 43.5, 0.0833, 862.8417, 0.0254, 3.889, 163.0, 240.0, 213.3555, -3.673006104957646,
             21.916179179999997, 26.644499999999994, 0.16346319018404903],
            [3.0, 135.0, 41.9063, 0.079, 690.3291, 0.0259, 3.8339999999999996, 167.0, 239.0, 213.7188,
             -3.653512310276645, 17.87952369, 25.281200000000013, 0.15138443113772462],
            [2.0, 126.0, 21.0586, 0.0406, 308.3583, 0.0684, 3.1702, 174.0, 240.0, 227.5859, -2.682382454353632,
             21.09170772, 12.41409999999999, 0.07134540229885052],
            [4.0, 197.0, 77.4805, 0.08900000000000001, 874.4709, 0.0243, 3.9442, 155.0, 239.0, 197.2773,
             -3.7172789286356345, 21.24964287, 41.7227, 0.26917870967741936],
            [7.0, 193.0, 88.8398, 0.0884, 810.1126, 0.0223, 3.9318, 150.0, 236.0, 186.0195, -3.803168600516064,
             18.065510980000003, 49.980500000000006, 0.33320333333333335]
        ],
        columns=['visible_mean', 'visible_max', 'visible_min', 'visible_mean_distribution',
                 'visible_contrast', 'visible_entropy', 'visible_second_angular_momentum',
                 'IR_mean', 'IR_max', 'IR_min', 'log_entropy', 'entropy_x_contrast',
                 'IR_range', 'IR_norm_range']
    )

    features_test = add_ir_norm_range(data)

    assert_frame_equal(features_true, features_test)


def test_add_ir_norm_range_unhappy():
    data = pd.DataFrame()
    with pytest.raises(AttributeError):
        add_ir_norm_range(data)
