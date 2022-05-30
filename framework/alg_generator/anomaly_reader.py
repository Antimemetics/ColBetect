import json
import os

from global_object.file import anomaly_users_file


def get_anomaly_users_file():
    for root, dirs, files in os.walk('data\\r4.2'):
        for file in files:
            with open(anomaly_users_file, 'a') as f:
                f.write(file[7: 14] + '\n')


def get_anomaly_users():
    anomaly = []
    with open(anomaly_users_file, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            anomaly.append(line)
    return anomaly
