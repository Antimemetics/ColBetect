import json
import sys
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp

from global_object.file import *
from global_object.weight import *
from framework.execution.ft_preparator import r_get_features
from framework.alg_generator.sorter import get_nodes_sorted_device


def r_load_whole_data():
    """
    adj：csr_matrix，shape=n*n
    features：lil_matrix，shape=n*ftd
    labels：numpy.ndarray，shape=n*lbl
    idx_train：list，range(0，tn)，idx=index
    idx_test：list，range(tn，n)
    """

    # 1 adj，csr_matrix
    # row、column、data
    nodes = get_nodes_sorted_device()
    for index in range(len(nodes)):
        nodes[index].index = index
    index_to_uid = {}
    uid_to_index = {}
    for node in nodes:
        index_to_uid[node.index] = node.uid
        uid_to_index[node.uid] = node.index
    node_num = len(nodes)

    row = []
    column = []
    data = []
    with open(device_edges_sorted, 'r') as f:
        for line in f:
            edge_dict = json.loads(line)
            row.append(uid_to_index[edge_dict['uid1']])
            column.append(uid_to_index[edge_dict['uid2']])
            data.append(edge_dict['weight'])

    adj_coo = sp.coo_matrix((data, (row, column)), shape=(node_num, node_num))
    adj = adj_coo.tocsr()

    # 2 features，lil_matrix
    # coo->lil
    features, anomaly_features = r_get_features(nodes)

    # 3 labels，numpy.ndarray
    # lbls = [lbl1, lbl2,..., lbln]
    normal = [1, 0]
    anomaly = [0, 1]
    lbls = []
    for node in nodes:
        if node.tag == 'anomaly':
            lbls.append(anomaly)
        else:
            lbls.append(normal)
    labels = np.array(lbls)

    nodes_cut = [int(node_num * train_size),
                 int(node_num * test_size)]

    # 4 idx_train：list，idx=index
    idx_train = range(0, nodes_cut[0])
    idx_test = range(nodes_cut[0], node_num)

    return adj, features, anomaly_features, labels, idx_train, idx_test,
