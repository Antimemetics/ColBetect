from framework.alg_generator.anomaly_reader import get_anomaly_users
from framework.alg_generator.edges_loader import load_edges_device
from framework.alg_generator.load_device import load_device
from framework.alg_generator.nodes_loader import load_nodes_device, \
    get_nodes_list_device
from framework.alg_generator.sorter import sort_nodes, \
    get_nodes_sorted_device
from global_object.file import *

reload = False
retrain = True

if reload:
    load_device()
    load_nodes_device()
    nodes = get_nodes_list_device()
    load_edges_device(nodes, device_edges)

    sort_nodes()
    nodes = get_nodes_sorted_device()
    load_edges_device(nodes, device_edges_sorted)
    print('reload done')

if retrain:
    exec(open('framework/execution/execute.py', encoding='utf-8').read())
