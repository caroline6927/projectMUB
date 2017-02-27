"""
1. Get features
    1) Autocorrelation analysis for feature selection
    2) Weekly and daily circadian rhythmicity of physical activity
2. Split data into train and test
3. Compute prediction accuracy
"""

# 1. get features

from __future__ import print_function
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as smtsa
from copy import deepcopy


# get feature method 1
def get_cr_features(data, var='date_time'):
    """
    Get time (minute count of the day), dayofweek and timeofweek (minute count of the week)
    :param data:
    :param var:
    :return:
    """
    # get new variables
    data['time'] = data[var].apply(lambda x: (x.hour * 60 + x.minute))
    data['dayofweek'] = data[var].apply(lambda x: x.dayofweek)
    data['timeofweek'] = data[var].apply(lambda x: (1439 * x.dayofweek + x.hour * 60 + x.minute))
    feature_lst = ['time', 'dayofweek', 'timeofweek']
    return data, feature_lst


# get feature method 2
def get_ac(data, var='steps'):
    # threshold: cut-off value for ac, n_obs = number of observations per day in data
    ac_results = smtsa.acf(data[var], nlags=len(data) // 2, unbiased=True, qstat=True, alpha=None)
    ac = ac_results[0][1:]
    ac_p = ac_results[2]
    ac_df = pd.DataFrame({'ac': ac, 'ac_p': ac_p})
    # print(ac_df.describe())
    return ac_df


def get_ac_features(data, ac_df, var='steps', threshold=99.8, cutoff=0.2, n_obs=95):
    feature_idx = []
    ths = np.percentile(ac_df.ac, threshold)
    for i in ac_df.index:
        if (ac_df.ac[i] >= max([ths, cutoff])) & (i >= n_obs):
            print('the %d th index has ac of %f' % (i, ac_df.ac[i]))
            feature_idx.append(i)
    print('feature indexes are %s' % str(feature_idx).strip('[]'))
    feature_lst = []

    for i in feature_idx:
        feature = []
        # fill in historical data as feature value
        for count in data.index:
            try:
                feature_value = data[var][count - i]
                feature.append(feature_value)
            except KeyError:
                feature.append(np.nan)
        key_name = var + 'feature_index' + str(i)
        # check proportion of missing value
        if (len(feature) - sum(np.isnan(feature))) >= 3000:
            print('index %d is valid' % i)
            data[key_name] = feature
            feature_lst.append(key_name)
        else:
            print('index %d is invalid' % i)
            # if (sum(np.isnan(feature)) / len(feature)) <= 0.1:
            #     print('index %d is valid' % i)
            #     data[key_name] = feature
            # else:
            #     print('index %d is invalid' % i)
    return data, feature_lst


def get_features(dt, method, params=[]):
    data = deepcopy(dt)
    if method == 'rhythm':
        if len(params) > 0:
            data, feature_lst = get_cr_features(data, var=params[0])
        else:
            data, feature_lst = get_cr_features(data)
    elif method == 'autoco':
        if len(params) > 0:
            feature_lst = []
            for var in params:
                data_ac_df = get_ac(data, var=params[0])
                data, feature_l = get_ac_features(data, data_ac_df, var=var)
                feature_lst.append(feature_l)
        else:
            data_ac_df = get_ac(data)
            data, feature_lst = get_ac_features(data, data_ac_df)
    data.dropna(inplace=True, how='any')
    data.reset_index(inplace=True, drop=True)
    return data, feature_lst
