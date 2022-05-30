import json
from global_object.file import *


def load_device():
    """raw data"""
    with open(device_json, 'w') as device_file:
        with open(raw_log_device, 'r') as f:
            f.readline()
            for line in f:
                para = line.split(',')
                '''
                0：id；1：date；2：user；3：pc；4：act
                '''
                date_whole = para[1]
                date = date_whole.split(' ')
                date = date[0]

                dict = {'id': para[0],
                        'date': date,
                        'user': para[2],
                        'pc': para[3],
                        'act': para[4].replace('\n', '')}

                json.dump(dict, device_file, ensure_ascii=False)
                device_file.write('\n')
