"""
New description
Process all new fitbit json files and update user fitbit database

- read json files
- generate csv files
- delete json files
- TODO: concate csv files to existing database
input: json files in a folder
output: csv file
"""


import pandas as pd
import glob
import configparser
import os
from datetime import datetime
import json

###############################
# configuration and constants #
###############################

config = configparser.ConfigParser()
config.read('user.ini')


def read_fitbit_json(i):
    fitbit_json = config.get('default', 'fitbit_json')
    fitbit_json_data = glob.glob(fitbit_json + "*.json")
    with open(fitbit_json_data[i]) as json_file:
        json_data = json.load(json_file)
        try:
            # heart
            value = pd.DataFrame.from_dict(json_data['activities-heart-intraday']['dataset'])
            date = json_data['activities-heart'][0]['dateTime']
            category = 'heart'
        except KeyError:
            # steps
            value = pd.DataFrame.from_dict(json_data['activities-steps-intraday']['dataset'])
            date = json_data['activities-steps'][0]['dateTime']
            category = 'steps'
    fitbit_df = value
    fitbit_df['date'] = date
    fitbit_df.rename(columns={'value': category}, inplace=True)
    return fitbit_df


def validate_fitbit_data(data, colname, cutoff=1.0):
    # If more than 80% of data reads in the json file are 0, the data may be incomplete.
    data_reads = sum(data[colname] == 0)/len(data)
    if data_reads >= cutoff:
        print("Missing data on %s. File ignored." % data['date'][0])
        return False
    else:
        return True


def categorize_fitbit_data():
    fitbit_json = config.get('default', 'fitbit_json')
    fitbit_json_data = glob.glob(fitbit_json + "*.json")
    heart_data = []
    steps_data = []
    for i in range(len(fitbit_json_data)):
        fitbit_df = read_fitbit_json(i)
        if 'heart' in list(fitbit_df.columns.values):
            if validate_fitbit_data(fitbit_df, 'heart'):
                heart_data.append(fitbit_df)
        if 'steps' in list(fitbit_df.columns.values):
            if validate_fitbit_data(fitbit_df, 'steps'):
                steps_data.append(fitbit_df)
    heart_data = pd.concat(heart_data)
    steps_data = pd.concat(steps_data)
    return heart_data, steps_data


def merge_fitbit_data():
    heart_data, step_data = categorize_fitbit_data()
    # merge heart and steps data
    # missing values are removed by setting merge(..., how='inner',...)
    fitbit_data = pd.merge(heart_data, step_data, how='inner', on=['date', 'time'])
    return fitbit_data


def write_fitbit_data():
    fitbit_data = merge_fitbit_data()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    fitbit_data.to_csv(config.get('default', 'fitbit_csv') + timestamp + 'fitbit.csv', encoding='ISO-8859-1')


def clean_cached_fitbit_json():
    fitbit_json = config.get('default', 'fitbit_json')
    fitbit_json_data = glob.glob(fitbit_json + "*.json")
    for file_ in fitbit_json_data:
        os.remove(file_)
