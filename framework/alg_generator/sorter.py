from global_object.file import *
from global_object.weight import *
from framework.graph.node import Node
from framework.alg_generator.nodes_loader import get_nodes_list_device
import json


def sort_nodes():
    nodes = get_nodes_list_device()
    normal_nodes = []
    anomaly_nodes = []
    for node in nodes:
        if node.tag == 'anomaly':
            anomaly_nodes.append(node)
        else:
            normal_nodes.append(node)

    train_node_num = int(len(nodes) * train_len)
    train_anomaly_num = int(len(anomaly_nodes) * train_len)
    uid = 0
    with open(device_nodes_sorted, 'w') as f:
        for i in range(train_anomaly_num):
            node = anomaly_nodes[i]
            node.uid = uid
            uid += 1
            _dump_node(node, f)
        for i in range(train_node_num - train_anomaly_num):
            node = normal_nodes[i]
            node.uid = uid
            uid += 1
            _dump_node(node, f)
        for i in range(train_anomaly_num, len(anomaly_nodes)):
            node = anomaly_nodes[i]
            node.uid = uid
            uid += 1
            _dump_node(node, f)
        for i in range(train_node_num - train_anomaly_num, len(normal_nodes)):
            node = normal_nodes[i]
            node.uid = uid
            uid += 1
            _dump_node(node, f)


def _dump_node(node, file):
    node_dict = node.get_node_dict()
    json.dump(node_dict, file, ensure_ascii=False)
    file.write('\n')


def get_nodes_sorted_device():
    nodes = []
    with open(device_nodes_sorted, 'r') as f:
        for line in f:
            node_dict = json.loads(line)
            node = Node(node_dict['uid'], node_dict['date'],
                        node_dict['user'], node_dict['tag'])
            node.pc_list = node_dict['pc_list']
            node.c_num = node_dict['c_num']
            node.d_num = node_dict['d_num']
            nodes.append(node)
    return nodes


def get_anomaly_uids():
    nodes = get_nodes_sorted_device()
    anomaly_uids = set()
    for i in range(int(len(nodes) * train_len)):
        node = nodes[i]
        if node.tag == 'anomaly':
            anomaly_uids.add(node.uid)
    return anomaly_uids
