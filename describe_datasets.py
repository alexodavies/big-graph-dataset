import argparse
import logging
import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import Compose
from tqdm import tqdm
from utils import better_to_nx, initialize_edge_weight
from datasets import get_train_datasets, get_val_datasets, get_test_datasets
from metrics import *




def get_metric_values(dataset):


    val_data = [better_to_nx(data)[0] for data in dataset]
    for i, item in enumerate(val_data):
        # item.remove_edges_from(nx.selfloop_edges(item))
        val_data[i] = clean_graph(item)

    # Metrics can be added here - should take an nx graph as input and return a numerical value
    metrics = [nx.number_of_nodes, nx.number_of_edges, safe_diameter,
               nx.average_clustering,] #, average_degree, ]
    metric_names = [prettify_metric_name(metric) for metric in metrics]
    # Compute metrics for all graphs
    metric_arrays = [np.array([metric(g) for g in tqdm(val_data, leave=False, desc=metric_names[i_metric])]) for i_metric, metric in enumerate(metrics)]

    return metric_arrays, metrics,  metric_names

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)

    # Get datasets
    my_transforms = Compose([initialize_edge_weight])

    train_datasets, train_names = get_train_datasets(my_transforms, num = args.num_train)
    val_datasets, val_names = get_val_datasets(my_transforms, num = args.num_val)
    test_datasets, _ = get_test_datasets(my_transforms, num = args.num_test)

    for i_dataset, dataset in enumerate(val_datasets):
        arrays, metrics, names = get_metric_values(dataset)
        if "ogbg" not in val_names[i_dataset]:
            continue
        print_string = f"{val_names[i_dataset]} & Val & {len(dataset)}"
        for i_name, name in enumerate(names):
            value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
            value = float('%.3g' % value)
            dev = float('%.3g' % dev)
            print_string += f"& {value} $\pm$ {dev}"
        print(print_string + r"\\")

    print("\n\n")



    for i_dataset, dataset in enumerate(test_datasets):
        arrays, metrics, names = get_metric_values(dataset)
        if "ogbg" not in val_names[i_dataset]:
            continue
        print_string = f"{val_names[i_dataset]} & Test & {len(dataset)}"
        for i_name, name in enumerate(names):
            value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
            value = float('%.3g' % value)
            dev = float('%.3g' % dev)
            print_string += f"& {value} $\pm$ {dev}"
        print(print_string + r"\\")

def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num_train', type=int, default=5000,
                        help='Number of points included in the train datasets')
    
    parser.add_argument('--num_val', type=int, default=1000,
                    help='Number of points included in the validation datasets')

    parser.add_argument('--num_test', type=int, default=1000,
                    help='Number of points included in the test datasets')
    
    parser.add_argument(
        '-s',
        '--score',
        action='store_true',
        help='Whether to compute similarity score against other datasets',
        default = False
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)

