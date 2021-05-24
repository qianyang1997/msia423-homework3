import yaml
import pandas as pd

import logging

from src.data_acquisition import save_data, load_data, logger0

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

logger.setLevel('INFO')
logger0.setLevel('INFO')

if __name__ == '__main__':

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    save_data(**config["save_data"])
    load_data(**config["load_data"])

'''
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df = pa.functionA(df, **config_pyA["functionA"])
df = pa.functionB(df, **config_pyA["functionB"])

if "save_results" in config_pyA:
    df.to_csv(config_pyA["save_results"])
'''