import json
import scipy.sparse as sp

from global_object.file import *
from global_object.weight import train_len
import pickle

"""mixed_nodes->ft"""

reload_features = False


def r_get_features(nodes):
    node_inedxes = set()
    ft_dim = 0
    if reload_features:
        _get_ft_in_number(nodes, device_ft_num)
        _get_ft_matrix_from_number(device_ft_num, device_ft_onehot)

    print('ready to load ft matrix')

    anomaly_row = []
    anomaly_col = []
    anomaly_data = []
    index_to_anomaly_index = {}

    anomaly_indexes = set()
    for i in range(int(len(nodes) * train_len)):
        node = nodes[i]
        if node.tag == 'anomaly':
            anomaly_indexes.add(node.index)

    print('anomaly indexed gotten')

    # line_count = 0
    # with open(device_ft_onehot, 'r') as f:
    #     while True:
    #         line = f.readline()
    #         if len(line) <= 1:
    #             break
    #
    #         line_count += 1
    #         if line_count % 100000 == 0:
    #             print(str(line_count) + ' done')
    #         index, ft_index, onehot_tag = line.split(' ')
    #         index = int(index)
    #         ft_index = int(ft_index)
    #         onehot_tag = int(onehot_tag)
    #
    #         node_inedxes.add(index)
    #         ft_dim = max(ft_dim, ft_index + 1)
    #
    #         row.append(index)
    #         column.append(ft_index)
    #         data.append(onehot_tag)
    #
    #         # 对异常的index，转写到异常矩阵中
    #         if index in anomaly_indexes:
    #             if index not in index_to_anomaly_index.keys():
    #                 index_to_anomaly_index[index] = len(index_to_anomaly_index)
    #
    #             anomaly_row.append(index_to_anomaly_index[index])
    #             anomaly_col.append(ft_index)
    #             anomaly_data.append(onehot_tag)

    with open(device_ft_onehot_row_pkl, 'rb') as f:
        row = pickle.load(f)
    with open(device_ft_onehot_col_pkl, 'rb') as f:
        column = pickle.load(f)
    with open(device_ft_onehot_data_pkl, 'rb') as f:
        data = pickle.load(f)

    for i in range(len(row)):
        index = row[i]

        node_inedxes.add(index)
        ft_dim = max(ft_dim, column[i] + 1)
        if i % 10000000 == 0:
            print(str(i) + ' done')

        if index in anomaly_indexes:
            if index not in index_to_anomaly_index.keys():
                index_to_anomaly_index[index] = len(index_to_anomaly_index)

            anomaly_row.append(index_to_anomaly_index[index])
            anomaly_col.append(column[i])
            anomaly_data.append(data[i])

    print('lists gotten')

    node_num = len(node_inedxes)
    print('node num: ' + str(node_num) + '; ft dim: ' + str(ft_dim))

    ft_coo = sp.coo_matrix((data, (row, column)), shape=(node_num, ft_dim))
    features = ft_coo.tolil()

    anomaly_num = len(anomaly_indexes)
    ft_anomalu_coo = sp.coo_matrix((anomaly_data, (anomaly_row, anomaly_col)),
                                   shape=(anomaly_num, ft_dim))
    anomaly_features = ft_anomalu_coo.tolil()
    return features, anomaly_features


def _get_ft_in_number(nodes, ft_file):
    """pc_list"""
    pc_to_index = {}
    with open(device_ft_num, 'w') as f:
        with open(ft_file, 'w') as ft_file_write:
            for node in nodes:
                pc_list = node.pc_list
                ft_pc = []
                for pc in pc_list:
                    if pc not in pc_to_index.keys():
                        pc_to_index[pc] = len(pc_to_index)
                    ft_pc.append(pc_to_index[pc])
                ft_pc.sort()

                ft = {'index': node.index,
                      'pc_list': ft_pc,
                      'c_num': node.c_num,
                      'd_num': node.d_num}
                json.dump(ft, ft_file_write, ensure_ascii=False)
                ft_file_write.write('\n')


def _get_ft_matrix_from_number(ft_num_file, ft_onehot_file):
    max_pc= 0
    with open(ft_num_file, 'r') as f:
        for line in f:
            ft = json.loads(line)
            node_max_pc = max(ft['pc_list'])
            max_pc = max(max_pc, node_max_pc)
    dim = max_pc + 2
    print('feature dims, pc: ' + str(max_pc) + ' + 2')

    row = []
    column = []
    data = []
    with open(ft_num_file, 'r') as f:
        for line in f:
            ft = json.loads(line)
            index = ft['index']
            node_pc_list = ft['pc_list']
            # pc_list
            base_index = 0
            for pc_index in range(max_pc):
                row.append(index)
                column.append(pc_index)
                if pc_index in node_pc_list:
                    data.append(1)
                else:
                    data.append(0)
            row.append(index)
            column.append(max_pc + 1)
            data.append(ft['c_num'])
            row.append(index)
            column.append(max_pc + 2)
            data.append(ft['d_num'])

    with open(ft_onehot_file, 'w') as f:
        for i in range(len(row)):
            f.write(str(row[i]) + ' ' + str(column[i]) + ' '
                    + str(data[i]) + '\n')

    with open(device_ft_onehot_row_pkl, 'wb') as f:
        pickle.dump(row, f)
    with open(device_ft_onehot_col_pkl, 'wb') as f:
        pickle.dump(column, f)
    with open(device_ft_onehot_data_pkl, 'wb') as f:
        pickle.dump(data, f)
