import time
import requests

import pandas as pd
import numpy as np

import logging

logger0 = logging.getLogger(__name__)


def save_data(html: str, attempt: int, wait: int) -> str:
    """Make html request to download cloud data.

    :param html: str - url to make html requests
    :param attempt: int - number of request attempts if connection error arises
    :param wait: int - number of seconds to wait before the next request attempt
    :return: str - downloaded text file
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
                response = ''
        except requests.exceptions.HTTPError:
            logger0.error("Invalid HTTP response. Check your url.")
            response = ''
            break
        else:
            break

    return response


def load_data(data: list, columns: list) -> pd.DataFrame:
    """Load cloud data from local path and write a concatenated csv.

    :param data: list - list of lists of lines of text in raw data text
    :param columns: list - columns to load
    :return: :obj: pandas dataframe - data as csv
    """
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

    return data
