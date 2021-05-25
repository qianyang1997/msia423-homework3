#!/usr/bin/env bash

# Acquire data from URL
python run.py acquire --config=config/config.yaml

# Read raw data and organize into csv
python run.py read --config=config/config.yaml --output=data/cloud.csv

# Generate features
python run.py featurize --input=data/cloud.csv --config=config/config.yaml --output=tests/data/featurized.csv

# Fit model and make predictions
python run.py train --input=tests/data/featurized.csv --config=config/config.yaml --output=tests/result/predictions.csv

# Evaluate model
python run.py evaluate --input=tests/result/predictions.csv