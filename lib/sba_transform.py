"""
This library contains basic transformation procedures for sedentary behavior analysis (SBA)
"""
from __future__ import print_function
import pandas as pd
import numpy as np


# step 1
def filter_na(data, target='heart'):
    """
    Filter out na values and 0 heart rate readings
    :param data:
    :param target:
    :return:
    """
    data.dropna(inplace=True, how='any', axis=0)
    mask = data[target].apply(lambda x: False if x == 0 else True)
    data = data[mask]
    data.reset_index(inplace=True, drop=True)
    return data


# step 2
def fill_missing(data, method='mean', timestamp_col='date_time'):
    data.sort_values(by=[timestamp_col], inplace=True)
    data.reset_index(inplace=True, drop=True)
    # get first date and last date
    start_day = data.loc[0, timestamp_col].strftime('%Y-%m-%d') + ' 00:00:00'
    end_day = data.loc[len(data) - 1, timestamp_col].strftime('%Y-%m-%d') + ' 23:45:00'
    time_stamps = pd.date_range(start=start_day, end=end_day, freq='15min')

    def match_data(data, n_data):
        for i in data.index:
            date_time_i = data.loc[i, timestamp_col]
            heart_i = data.loc[i, 'heart']
            steps_i = data.loc[i, 'steps']
            n_data.loc[n_data['date_time'] == date_time_i, 'heart'] = heart_i
            n_data.loc[n_data['date_time'] == date_time_i, 'steps'] = steps_i
        return n_data

    np.random.seed(0)
    if method == 'mean':
        n_data = pd.DataFrame({'date_time': time_stamps,
                               'heart': np.mean(data.heart),
                               'steps': np.mean(data.steps)})
        n_data = match_data(data, n_data)
    elif method == 'rndminmax':
        n_data = pd.DataFrame({'date_time': time_stamps,
                               'heart': np.mean(data.heart),
                               'steps': np.random.randint(min(data.steps), max(data.steps), len(time_stamps))})
        n_data = match_data(data, n_data)
    elif method == 'rnd50zeromax':
        n_data = pd.DataFrame({'date_time': time_stamps,
                               'heart': np.mean(data.heart),
                               'steps': max(data.steps) * np.random.randint(0, 2, len(time_stamps))})
        n_data = match_data(data, n_data)
    elif method == 'zero':
        n_data = pd.DataFrame({'date_time': time_stamps,
                               'heart': np.mean(data.heart),
                               'steps': 0})
        n_data = match_data(data, n_data)
    elif method == 'max':
        n_data = pd.DataFrame({'date_time': time_stamps,
                               'heart': np.mean(data.heart),
                               'steps': max(data.steps)})
        n_data = match_data(data, n_data)
    elif method == 'none':
        n_data = data
    return n_data


# step 3
def get_inactive(data, target='steps'):
    # TODO: consider changing to positive for active
    """
    Get inactive indicator (postive for inactive)
    :param data:
    :param target:
    :return:
    """
    data['inactive'] = data[target].apply(lambda x: 10 if x == 0 else -10)
    return data


def filter_sleeping_time(data, target='date_time'):
    """
    Filter out sleeping time
    :param data:
    :param target:
    :return:
    """
    mask = data[target].apply(lambda x: False if x.hour in range(7) else True)
    data = data[mask]
    data.reset_index(inplace=True, drop=True)
    return data


def sba_pipeline(data, without_sleep=False):
    data = get_inactive(data)
    if without_sleep:
        data = filter_sleeping_time(data)
    return data



