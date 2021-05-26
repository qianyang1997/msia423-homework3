# Acquire data from URL
python3 run.py acquire --config=config/config.yaml --output=data/cloud.data

# Read raw data and organize into csv
python3 run.py read --input=data/cloud.data --config=config/config.yaml --output=data/cloud.csv

# Generate features
python3 run.py featurize --input=data/cloud.csv --config=config/config.yaml --output=data/featurized.csv

# Fit model and make predictions
python3 run.py train --input=data/featurized.csv --config=config/config.yaml --output=data/predictions.csv

# Evaluate model
python3 run.py evaluate --input=data/predictions.csv