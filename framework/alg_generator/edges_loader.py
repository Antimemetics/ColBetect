import json
from global_object.weight import *
from framework.graph.edge import Edge


def load_edges_device(nodes, edges_file):
    # 1
    edges = []

    # 2
    _add_chronological(nodes, edges)
    # 3
    _add_same_user(nodes, edges)
    # 4
    _add_same_pc(nodes, edges)

    # dump
    with open(edges_file, 'w') as f:
        for edge in edges:
            edge_dict = edge.get_edge_dict()
            json.dump(edge_dict, f, ensure_ascii=False)
            f.write('\n')


def _add_chronological(nodes, edges):
    for i in range(len(nodes) - 1):
        node1 = nodes[i]
        node2 = nodes[i + 1]
        edge = Edge(node1.uid, node2.uid, chronological_weight)
        edges.append(edge)


def _add_same_user(nodes, edges):
    user_to_nodes = {}
    for node in nodes:
        user = node.user
        if user not in user_to_nodes.keys():
            user_to_nodes[user] = []
        user_to_nodes[user].append(node)
    for user in user_to_nodes.keys():
        _add_chronological(user_to_nodes[user], edges)


def _add_same_pc(nodes, edges):
    pc_to_nodes = {}
    for node in nodes:
        for pc in node.pc_list:
            if pc not in pc_to_nodes.keys():
                pc_to_nodes[pc] = []
            pc_to_nodes[pc].append(node)
    for pc in pc_to_nodes.keys():
        _add_chronological(pc_to_nodes[pc], edges)
