import os
import pandas as pd


# get file names
def get_file_log(rootdir, postfix='.csv'):
    file_log = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_name = os.path.join(subdir, file)
            if postfix in file_name:
                file_log.append(file_name)
    return file_log


# get file data
def read_csv_from_filename(filename):
    data = pd.read_csv(filename, parse_dates=['date_time'], dayfirst=True)
    return data


def read_csv_from_log(file_log):
    data_ = []
    for f in file_log:
        data = read_csv_from_filename(f)
        data_.append(data)
    df = pd.concat(data_)
    df.sort_values(by=['date_time'], inplace=True, dayfirst=True)   # TODO: dayfirst?
