import json
import pickle

import torch
import numpy as np
import scipy.sparse as sp

from framework.execution import data_preparer
from global_object.file import device_ft_onehot_row_pkl, \
    device_ft_onehot_col_pkl, device_ft_onehot_data_pkl, device_edges_sorted
from global_object.weight import train_len
from framework.alg_generator.sorter import get_nodes_sorted_device, \
    get_anomaly_uids


def get_neg_para(anomaly_features):
    anomaly_uids = get_anomaly_uids()

    # 1 n
    neg_num = len(anomaly_uids)

    # 2 ft
    anomaly_features, anomaly__ = data_preparer.preprocess_features(
        anomaly_features)
    seq3 = torch.FloatTensor(anomaly_features[np.newaxis])

    # 3 matrix
    row = []
    column = []
    data = []

    uid_to_index = {}
    for uid in anomaly_uids:
        if uid not in uid_to_index.keys():
            uid_to_index[uid] = len(uid_to_index)

    with open(device_edges_sorted, 'r', encoding='utf-8') as f:
        for line in f:
            edge_dict = json.loads(line)
            uid1 = edge_dict['uid1']
            uid2 = edge_dict['uid2']
            if uid1 in anomaly_uids and uid2 in anomaly_uids:
                row.append(uid_to_index[uid1])
                column.append(uid_to_index[uid2])
                data.append(edge_dict['weight'])

    adj3_coo = sp.coo_matrix((data, (row, column)), shape=(neg_num, neg_num))
    adj3 = adj3_coo.tocsr()
    adj3 = data_preparer.normalize_adj(adj3 + sp.eye(adj3.shape[0]))
    adj3 = data_preparer.sparse_mx_to_torch_sparse_tensor(adj3)

    return neg_num, adj3, seq3
