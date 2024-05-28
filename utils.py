import random
import networkx as nx
import numpy as np
import torch
from littleballoffur.exploration_sampling import *
from tqdm import tqdm
import concurrent.futures
import os
from itertools import islice, chain
import itertools
# from metrics import get_metric_values
import concurrent.futures
import networkx as nx
import numpy as np
from tqdm import tqdm

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

class ComponentSlicer:
    def __init__(self, comp_1 = 0, comp_2 = 1):
        self.comp_1 = comp_1
        self.comp_2 = comp_2

    def fit(self, X):
        pass

    def transform(self, X):
        return np.concatenate((X[:, self.comp_1].reshape(-1,1), X[:, self.comp_2].reshape(-1,1)), axis=1)

def average_degree(g):
    if nx.number_of_edges(g)/nx.number_of_nodes(g) < 1:
        print(g)
    return nx.number_of_edges(g)/nx.number_of_nodes(g)
#
def safe_diameter(g):
    """
    Returns either the diameter of a graph or -1 if it has multiple components
    Args:
        g: networkx.Graph

    Returns:
        either the diameter of the graph or -1
    """
    try:
        return nx.diameter(g)
    except:
        return -1

def three_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 3)
    return len(list(cycles))

def four_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 4)
    return len(list(cycles))

def five_cycle_worker(g):
    """
    Returns the number of 5-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 5)
    return len(list(cycles)) / nx.number_of_nodes(g)

def five_cycles(graphs):
    """
    Returns the number of 5-cycles per graph in a list of graphs
    Args:
        graphs: list of networkx.Graph objects

    Returns:
        list of 5-cycle counts
    """
    sample_ref = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        for n_coms in tqdm(executor.map(five_cycle_worker, graphs), desc="Five cycles"):
            sample_ref.append(n_coms)

    return sample_ref


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def initialize_edge_weight(data):
	data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
	return data

def better_to_nx(data):
    """
    Converts a pytorch_geometric.data.Data object to a networkx graph,
    robust to nodes with no edges, unlike the original pytorch_geometric version

    Args:
        data: pytorch_geometric.data.Data object

    Returns:
        g: a networkx.Graph graph
        labels: torch.Tensor of node labels
    """
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    for ilabel in range(labels.shape[0]):
        if ilabel not in np.unique(edges):
            g.add_node(ilabel)

    return g, labels

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def prettify_metric_name(metric):
    try:
        metric_name = str(metric).split(' ')[1]
    except:
        metric_name = str(metric)
    # metrics = [nx.number_of_nodes, nx.number_of_edges, nx.density, safe_diameter,
    #            average_degree, nx.average_clustering, nx.transitivity]
    pretty_dict = {"number_of_nodes": "Num. Nodes",
                   "number_of_edges": "Num. Edges",
                   "density": "Density",
                   "safe_diameter": "Diameter",
                   "average_degree": "Avg. Degree",
                   "average_clustering": "Avg. Clust.",
                   "transitivity": "Trans.",
                   "three_cycles": "Num. 3-Cycles",
                   "four_cycles":"Num. 4-Cycles"}

    return pretty_dict[metric_name]

def clean_graph(g):
    Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(Gcc[0]).copy()
    g.remove_edges_from(nx.selfloop_edges(g))

    return g


# def ESWR(graph, n_graphs, size):
#     possible_samplers = [MetropolisHastingsRandomWalkSampler, DiffusionSampler, ForestFireSampler]
#     sampler_list = []
#     for sampler in possible_samplers:
#         for i in range(24,96):
#             sampler_list.append(sampler(i))

#     graphs = []
#     for i in tqdm(range(n_graphs), desc = "Sampling from large graph"):
#         sampler = sampler_list[np.random.randint(len(sampler_list))]
#         g = nx.convert_node_labels_to_integers(sampler.sample(graph))
#         graphs.append(g)

#     return graphs

# def sample_graph(sampler, graph):
#     return nx.convert_node_labels_to_integers(sampler.sample(graph))

# def ESWR(graph, n_graphs, size):
#     possible_samplers = [MetropolisHastingsRandomWalkSampler, DiffusionSampler, ForestFireSampler]
#     sampler_list = [sampler(i) for sampler in possible_samplers for i in range(24, 96)]

#     graphs = []
#     with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() // 2) as executor:
#         futures = [executor.submit(sample_graph, np.random.choice(sampler_list), graph) for _ in range(n_graphs)]
#         for future in tqdm(concurrent.futures.as_completed(futures), total=n_graphs, desc="Sampling from large graph"):
#             graphs.append(future.result())

#     return graphs



def sample_graph(sampler, graph):
    return nx.convert_node_labels_to_integers(sampler.sample(graph))

def chunked_iterable(iterable, size):
    """Helper function to split an iterable into chunks of a given size."""
    it = iter(iterable)
    for first in it:
        yield list(islice(chain([first], it), size - 1))

def process_chunk(chunk, sampler_list, graph):
    chunk_graphs = []
    for _ in chunk:
        sampler = np.random.choice(sampler_list)
        g = sample_graph(sampler, graph)
        chunk_graphs.append(g)
    return chunk_graphs

def ESWR(graph, n_graphs, size):
    possible_samplers = [MetropolisHastingsRandomWalkSampler, DiffusionSampler, ForestFireSampler]
    sampler_list = [sampler(i) for sampler in possible_samplers for i in range(24, 96)]

    max_workers = os.cpu_count() // 2  # Use half the available CPU cores

    # Chunk the tasks to reduce the overhead
    chunk_size = n_graphs // max_workers or 1
    print(f"\nSampling {n_graphs} in {max_workers} chunks with size {chunk_size}")
    graph_chunks = list(chunked_iterable(range(n_graphs), chunk_size))

    graphs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, sampler_list, graph) for chunk in graph_chunks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Sampling from large graph"):
            graphs.extend(future.result())
    print("Done sampling!\n")
    return graphs

def describe_one_dataset(dataset):
    print(r"Stage  &  Num  &  X shape  &  E shape  &  Y shape  &  Num. Nodes  &  Num. Edges  &  Diameter  &  Clustering \\")
    print_string = f"{dataset.stage} & {len(dataset)}"
    arrays, metrics, names = get_metric_values(dataset)
    one_sample = dataset[0]

    x_shape, edge_shape, y_shape = one_sample.x, one_sample.edge_attr, one_sample.y
    x_shape = x_shape.shape[1] if x_shape is not None else "none" 
    edge_shape = edge_shape.shape[1] if edge_shape is not None else "none" 
    try:
        y_shape = y_shape.shape if y_shape is not None else "none" 
        if y_shape != "none":
            if len(y_shape) == 2:
                y_shape = y_shape[1]
            else:
                y_shape = y_shape[0]
    except:
        y_shape = "none"

    print_string += f" & {x_shape} & {edge_shape} & {y_shape} "
    for i_name, name in enumerate(names):
        value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
        value = float('%.3g' % value)
        dev = float('%.3g' % dev)
        print_string += f"& {value} $\pm$ {dev}"

    print(print_string + r"\\")
