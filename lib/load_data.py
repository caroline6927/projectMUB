import os


def get_file_log(rootdir, postfix='.csv'):
    file_log = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_name = os.path.join(subdir, file)
            if postfix in file_name:
                file_log.append(file_name)
    return file_log

