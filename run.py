import argparse
import logging

import yaml
import pandas as pd

from src.data_acquisition import save_data, load_data
from src.feature_generation import feature_gen, add_ir_norm_range,\
    add_ir_range, add_log_entropy, add_entropy_x_contrast
from src.train_model import train_test_split, fit_model
from src.evaluate_model import evaluation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level='INFO')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="load data, create features, and run models on cloud data")

    parser.add_argument('step', help='Which step to run', choices=['acquire', 'read', 'featurize', 'train', 'evaluate'])
    parser.add_argument('--input', '-i', default=None, help='input filepath')
    parser.add_argument('--config', default='config/config.yaml', help='path to config yaml file')
    parser.add_argument('--output', '-o', default=None, help='output filepath')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # save data (to raw text file)
    if args.step == 'acquire':
        output = save_data(**config["save_data"])

    # read data
    elif args.step == 'read':
        with open(args.input, 'r') as f:
            data = [[s for s in line.split(' ') if s != ''] for line in f.readlines()]
        output = load_data(data, **config["load_data"])

    # feature generation
    elif args.step == 'featurize':
        data = pd.read_csv(args.input)
        feature_config = config["generate_features"]

        # users have the option to specify output paths and download granular feature files
        features, labels = feature_gen(data)
        if "features_output_path" in feature_config["feature_gen"] and \
           "labels_output_path" in feature_config["feature_gen"]:
            features.to_csv(feature_config["feature_gen"]["features_output_path"], index=False)
            labels.to_csv(feature_config["feature_gen"]["labels_output_path"], index=False)

        features = add_log_entropy(features)
        if "output_path" in feature_config["add_log_entropy"]:
            features.to_csv(feature_config["add_log_entropy"]["output_path"], index=False)

        features = add_entropy_x_contrast(features)
        if "output_path" in feature_config["add_entropy_x_contrast"]:
            features.to_csv(feature_config["add_entropy_x_contrast"]["output_path"], index=False)

        features = add_ir_range(features)
        if "output_path" in feature_config["add_ir_range"]:
            features.to_csv(feature_config["add_ir_range"]["output_path"], index=False)

        features = add_ir_norm_range(features)
        if "output_path" in feature_config["add_ir_norm_range"]:
            features.to_csv(feature_config["add_ir_norm_range"]["output_path"], index=False)

        output = pd.concat([features, labels], axis=1)

    # model training
    elif args.step == 'train':
        data = pd.read_csv(args.input)
        model_config = config["train_model"]
        features, labels = feature_gen(data)
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            **model_config['train_test_split'])
        pred = fit_model(X_train, y_train, X_test, **model_config['fit_model'], **model_config['model_params'])

        output = pd.concat([pred.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # model evaluation
    elif args.step == 'evaluate':
        data = pd.read_csv(args.input)
        pred, y_test = feature_gen(data)
        evaluation(y_test, pred)
        output = None

    if args.output is not None:
        if type(output) == str and output != '':
            with open(args.output, 'w') as text:
                text.write(output)
                logger.info(f'Output saved to {args.output}.')
        elif type(output) == pd.DataFrame:
            output.to_csv(args.output, index=False)
            logger.info(f'Output saved to {args.output}.')
        else:
            logger.error(f'Error: Output in questionable format.')
