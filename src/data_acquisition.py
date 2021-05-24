import yaml
import time
import requests

import pandas as pd
import numpy as np

import logging

logger0 = logging.getLogger(__name__)


def save_data(html: str, local_path: str, attempt: int, wait: int) -> None:
    """Make html request to download cloud data.

    :param html: str - url to make html requests
    :param local_path: str - local path to save raw data file
    :param attempt: int - number of request attempts if connection error arises
    :param wait: int - number of seconds to wait before the next request attempt
    :return: None
    """
    for i in range(attempt):
        try:
            response = requests.get(html).text
        except requests.exceptions.ConnectionError:
            if i + 1 < attempt:
                logger0.warning(f"There was a connection error during attempt {i} of {attempt}. "
                                f"Waiting {wait} seconds then trying again.")
                time.sleep(wait)
            else:
                logger0.error("Max attempt reached. There was a connection error when attempting to call %s.", html)
        except requests.exceptions.HTTPError:
            logger0.error("Invalid HTTP response. Check your url.")
            break
        else:
            with open(local_path, 'w') as text:
                text.write(response)
            logger0.info(f"Data file downloaded to {local_path}.")
            break


def load_data(input_path: str, output_path: str, columns: list) -> pd.DataFrame:
    """Load cloud data from local path and write a concatenated csv.

    :param input_path: str - local data input file path
    :param output_path: str - local data output file path
    :param columns: list - columns to load
    :return: :obj: pandas dataframe - data as csv
    """
    with open(input_path, 'r') as f:
        data = [[s for s in line.split(' ') if s != ''] for line in f.readlines()]

    # extract data for first cloud
    first_cloud = data[53:1077]
    first_cloud = [[float(s.replace('/n', '')) for s in cloud]
                   for cloud in first_cloud]
    first_cloud = pd.DataFrame(first_cloud, columns=columns)
    first_cloud['class'] = np.zeros(len(first_cloud))

    # extract data for second cloud
    second_cloud = data[1082:2105]
    second_cloud = [[float(s.replace('/n', '')) for s in cloud]
                    for cloud in second_cloud]
    second_cloud = pd.DataFrame(second_cloud, columns=columns)
    second_cloud['class'] = np.ones(len(second_cloud))

    # concatenate dataframes for training
    data = pd.concat([first_cloud, second_cloud])

    data.to_csv(output_path, index=False)
    logger0.info(f'data successfully written to {output_path}.')

    return data