import yaml
import pandas as pd

import logging

from src.data_acquisition import save_data, load_data, logger0
from src.feature_generation import feature_gen, add_ir_norm_range,\
    add_ir_range, add_log_entropy, add_entropy_x_contrast, logger1
from src.train_model import train_test_split, fit_model, logger2
from src.evaluate_model import evaluation, logger3

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

logger.setLevel('INFO')
logger0.setLevel('INFO')
logger1.setLevel('INFO')
logger2.setLevel('INFO')
logger3.setLevel('INFO')


if __name__ == '__main__':

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # save data
    save_data(**config["save_data"])

    # feature generation
    feature_config = config["generate_features"]

    data = load_data(**config["load_data"])
    features, labels = feature_gen(data, **feature_config["feature_gen"])
    features = add_log_entropy(features, **feature_config["add_log_entropy"])
    features = add_entropy_x_contrast(features, **feature_config["add_entropy_x_contrast"])
    features = add_ir_range(features, **feature_config["add_ir_range"])
    features = add_ir_norm_range(features, **feature_config["add_ir_norm_range"])

    # model training
    model_config = config["train_model"]
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        **model_config['train_test_split'])
    pred = fit_model(X_train, y_train, X_test, model_config['features'], **model_config['model_params'])

    # model evaluation
    evaluation(y_test, pred, **config["evaluate_model"])

'''
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df = pa.functionA(df, **config_pyA["functionA"])
df = pa.functionB(df, **config_pyA["functionB"])

if "save_results" in config_pyA:
    df.to_csv(config_pyA["save_results"])
'''