from global_object.file import *
import json

from framework.graph.node import Node
from framework.alg_generator.anomaly_reader import get_anomaly_users


def load_nodes_device():
    """get nodes"""
    date_num = -1
    uid = 0
    now_date = None
    nodes = []
    today_users_to_nodes = {}
    first_day = True
    anomaly_users = get_anomaly_users()

    with open(device_nodes, 'w') as f:
        pass

    with open(device_json, 'r') as device_file:

        for line in device_file:
            log_line = json.loads(line)
            # date
            if log_line['date'] != now_date and not first_day:
                with open(device_nodes, 'a') as nodes_file:
                    for node in nodes:
                        node_dict = node.get_node_dict()
                        json.dump(node_dict, nodes_file, ensure_ascii=False)
                        nodes_file.write('\n')
                # clear
                date_num += 1
                now_date = log_line['date']
                today_users_to_nodes = {}
                nodes = []
            first_day = False

            # user
            user = log_line['user']
            if log_line['user'] not in today_users_to_nodes.keys():
                tag = 'normal'
                if user in anomaly_users:
                    tag = 'anomaly'
                node = Node(uid, date_num, user, tag)
                uid += 1
                today_users_to_nodes[user] = node
                nodes.append(node)
            # add node
            node = today_users_to_nodes[user]
            node.add_line_dict(log_line)


def get_nodes_list_device():
    nodes = []
    with open(device_nodes, 'r') as f:
        for line in f:
            node_dict = json.loads(line)
            node = Node(node_dict['uid'], node_dict['date'],
                        node_dict['user'], node_dict['tag'])
            node.pc_list = node_dict['pc_list']
            node.c_num = node_dict['c_num']
            node.d_num = node_dict['d_num']
            nodes.append(node)
    return nodes
