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
