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
import matplotlib.pyplot as plt
from torch.utils.data.dataset import ConcatDataset
from collections import deque
from typing import Union

import networkit as nk

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


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


# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

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

    if data.y is None or len(data.y.shape) < 2:
        if data.x is not None:
            labels = data.x[:,0].cpu().numpy()
        else:
            labels = np.ones(data.num_nodes)

    elif torch.sum(torch.unique(data.y)) == 1:
        labels = torch.argmax(data.y, dim = 1).cpu().numpy()

    else:
        # Some tasks are edge classification/regression
        if data.y.shape[0] == data.num_nodes:
            labels = data.y.cpu().numpy()
        else:
            labels = np.ones(data.num_nodes)

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
    try:
        return pretty_dict[metric_name]
    except:
        print(f"Couldn't find {metric} to make pretty!\n Valid options are:\n{pretty_dict}")

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

def normalize_l2(input_tensor):
    """
    Normalize an (N x D) tensor with the L2 norm.

    Parameters:
    input_tensor (torch.Tensor): Input tensor of shape (N, D)

    Returns:
    torch.Tensor: Normalized tensor of shape (N, D)
    """
    # Uncomment to actually normalize
        # # Compute the L2 norm for each vector
        # l2_norm = torch.linalg.vector_norm(input_tensor, ord=2, dim=1, keepdim=True)
        
        # # Avoid division by zero by creating a mask
        # l2_norm = l2_norm + (l2_norm == 0) * 1e-10
        
        # # Normalize the input tensor
        # normalized_tensor = input_tensor / l2_norm
        
        # return normalized_tensor

    return input_tensor

def wandb_cfg_to_actual_cfg(original_cfg, wandb_cfg):
    """
    Retrive wandb config from saved file
    Args:
        original_cfg: the config from this run
        wandb_cfg: the saved config from the training run

    Returns:
        a config with values updated to those from the saved training run
    """
    original_keys = list(vars(original_cfg).keys())
    wandb_keys = list(wandb_cfg.keys())

    for key in original_keys:
        if key not in wandb_keys:
            continue

        vars(original_cfg)[key] = wandb_cfg[key]['value']

    return original_cfg


# Please note: this is a workaround for littleballoffur with Python>=3.11
# Error thrown without:
# TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).

class CustomDiffusionSampler(DiffusionSampler):
    def _create_initial_node_set(self, graph, start_node):
        """
        Choosing an initial node.
        """
        self._sampled_edges = []
        if start_node is not None:
            if start_node >= 0 and start_node < self.backend.get_number_of_nodes(graph):
                self._sampled_nodes = list(set([start_node]))
            else:
                raise ValueError("Starting node index is out of range.")
        else:
            node = random.choice(range(self.backend.get_number_of_nodes(graph)))
            self._sampled_nodes = list(set([node]))

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        source_node = random.sample(self._sampled_nodes, 1)[0]
        neighbor = self.backend.get_random_neighbor(graph, source_node)
        if neighbor not in self._sampled_nodes:
            self._sampled_nodes.append(neighbor)
            self._sampled_edges.append([source_node, neighbor])
            self._sampled_edges.append([neighbor, source_node])

    def sample(
        self, graph: Union[NXGraph, NKGraph], start_node: int = None
    ) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes with a diffusion process.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
            * **start_node** *(int, optional)* - The start node.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_node_set(graph, start_node)
        while len(self._sampled_nodes) < self.number_of_nodes:
            self._do_a_step(graph)
        new_graph = self.backend.get_subgraph(graph, list(self._sampled_nodes))
        return new_graph

class CustomForestFireSampler(ForestFireSampler):

    def _create_node_sets(self, graph):
        """
        Create a starting set of nodes.
        """
        self._sampled_nodes = set()
        self._set_of_nodes = set(range(self.backend.get_number_of_nodes(graph)))
        self._visited_nodes = deque(maxlen=self.max_visited_nodes_backlog)

    def _start_a_fire(self, graph):
        """
        Starting a forest fire from a single node.
        """
        if not self._sampled_nodes:
            # If this is the first fire, select a random seed node
            seed_node = random.choice(tuple(self._set_of_nodes))
            node_queue = deque([seed_node])
        else:
            # Use unvisited nodes as candidates for the next fire
            remaining_nodes = list(self._set_of_nodes - self._sampled_nodes)
            if not remaining_nodes:
                print("Warning: No remaining nodes to sample.")
                return
            seed_node = random.choice(remaining_nodes)
            node_queue = deque([seed_node])

        self._sampled_nodes.add(seed_node)

        while len(self._sampled_nodes) < self.number_of_nodes:
            if not node_queue:
                # Restart the fire if needed
                if not self._visited_nodes:
                    print(
                        "Warning: could not collect the required number of nodes. The fire could not find enough nodes to burn."
                    )
                    break
                node_queue.extend(
                    self._visited_nodes.popleft()
                    for _ in range(min(self.restart_hop_size, len(self._visited_nodes)))
                )
                continue

            top_node = node_queue.popleft()
            neighbors = self.backend.get_neighbors(graph, top_node)
            unvisited_neighbors = [n for n in neighbors if n not in self._sampled_nodes]

            if unvisited_neighbors:
                score = np.random.geometric(self.p)
                count = min(len(unvisited_neighbors), score)
                burned_neighbors = random.sample(unvisited_neighbors, count)

                self._sampled_nodes.update(burned_neighbors)
                self._visited_nodes.extendleft(
                    n for n in unvisited_neighbors if n not in burned_neighbors
                )

                node_queue.extend(burned_neighbors)

    def sample(self, graph: Union[NXGraph, NKGraph]) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes iteratively with a forest fire sampler.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_node_sets(graph)
        while len(self._sampled_nodes) < self.number_of_nodes:
            self._start_a_fire(graph)
        new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
        return new_graph


class CustomMetropolisHastingsRandomWalkSampler(MetropolisHastingsRandomWalkSampler):
    # def _do_a_step(self, graph):
    #     # Ensure that self._sampled_nodes is a list before sampling
    #     if isinstance(self._sampled_nodes, (set, dict)):
    #         self._sampled_nodes = list(self._sampled_nodes)
    #     super()._do_a_step(graph)

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        score = random.uniform(0, 1)
        new_node = self.backend.get_random_neighbor(graph, self._current_node)
        ratio = float(self.backend.get_degree(graph, self._current_node)) / float(
            self.backend.get_degree(graph, new_node)
        )
        ratio = ratio ** self.alpha
        if score < ratio:
            self._current_node = new_node
            self._sampled_nodes.add(self._current_node)

def sample_graph(sampler, graph):
    graph = nx.convert_node_labels_to_integers(graph)
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
    possible_samplers = [CustomMetropolisHastingsRandomWalkSampler, CustomDiffusionSampler, CustomForestFireSampler]
    sampler_list = [sampler(i) for sampler in possible_samplers for i in range(24, size)]

    max_workers = min(os.cpu_count() // 2, 8)  # Use a reasonable number of threads

    # Chunk the tasks to reduce the overhead
    chunk_size = n_graphs // max_workers or 1
    print(f"\nSampling {n_graphs} in {max_workers} chunks with size {chunk_size} from {graph}")
    graph_chunks = list(chunked_iterable(range(n_graphs), chunk_size))

    graphs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, sampler_list, graph) for chunk in graph_chunks]
        # for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Sampling from {graph}"):
        for future in concurrent.futures.as_completed(futures):
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
        print_string += f"& {value} $\pm$ {dev} "

    print(print_string + r"\\")

def vis_networkx(g, filename = None):
    """
    Visualise a networkx.Graph object
    Args:
        g: networkx.Graph object
        filename: if passed, this is the filename for the saved image. If not passed, calls plt.show()

    Returns:

    """
    fig, ax = plt.subplots(figsize = (2,2))

    # if "ogbg" not in filename:
    pos = nx.spring_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    nx.draw_networkx_nodes(g, pos=pos, node_color=np.arange(g.number_of_nodes()),
                            edgecolors="black",
                            cmap="Dark2", node_size=5,
                            vmin=0, vmax=g.number_of_nodes(), ax=ax)
    # else:
    #     im = vis_molecule(nx_to_rdkit(g, labels))
    #     ax.imshow(im)

    ax.axis('off')
    # ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi = 300)
        plt.close()

    plt.close()

def vis_from_pyg(data, filename = None, ax = None, save = True):
    """
    Visualise a pytorch_geometric.data.Data object
    Args:
        data: pytorch_geometric.data.Data object
        filename: if passed, this is the filename for the saved image. Ignored if ax is not None
        ax: matplotlib axis object, which is returned if passed

    Returns:

    """
    g, labels = better_to_nx(data)
    if ax is None:
        fig, ax = plt.subplots(figsize = (2,2))
        ax_was_none = True
    else:
        ax_was_none = False

    # if "ogbg" not in filename:
    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    if np.unique(labels).shape[0] != 1:
        nx.draw_networkx_nodes(g, pos=pos, node_color=labels,
                                edgecolors="black",
                                cmap="Dark2", node_size=64,
                                vmin=0, vmax=10, ax=ax)
    # else:
    #     im = vis_molecule(nx_to_rdkit(g, labels))
    #     ax.imshow(im)

    ax.axis('off')
    # ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()

    if not ax_was_none:
        return ax
    elif filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi = 300)
        plt.close()

    plt.close()

def lowres_vis_from_pyg(data, filename = None, ax = None, save = True):
    """
    Visualise a pytorch_geometric.data.Data object
    Args:
        data: pytorch_geometric.data.Data object
        filename: if passed, this is the filename for the saved image. Ignored if ax is not None
        ax: matplotlib axis object, which is returned if passed

    Returns:

    """
    g, labels = better_to_nx(data)
    if ax is None:
        fig, ax = plt.subplots(figsize = (2,2))
        ax_was_none = True
    else:
        ax_was_none = False

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    if np.unique(labels).shape[0] != 1:
        nx.draw_networkx_nodes(g, pos=pos, node_color=labels,
                                edgecolors="black",
                                cmap="Dark2", node_size=64,
                                vmin=0, vmax=10, ax=ax)


    ax.axis('off')
    # ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()

    if not ax_was_none:
        return ax
    elif filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi = 50)
        plt.close()

    plt.close()



def vis_grid(datalist, filename, show_plot = False):
    """
    Visualise a set of graphs, from pytorch_geometric.data.Data objects
    Args:
        datalist: list of pyg.data.Data objects
        filename: the visualised grid is saved to this path

    Returns:
        None
    """

    new_data_list = []
    if type(datalist) == ConcatDataset:
        print("Found concat dataset in ToP conversion")
        for dataset in datalist.datasets:
            for item in dataset:
                new_data_list.append(item)
        datalist = new_data_list

    # Trim to square root to ensure square grid
    grid_dim = int(np.sqrt(len(datalist)))

    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(8,8))

    # Unpack axes
    axes = [num for sublist in axes for num in sublist]


    try:
        example = datalist[0].x
    except:
        datalist = datalist.datasets[0]

    for i_axis, ax in enumerate(axes):
        ax = vis_from_pyg(datalist[i_axis], ax = ax, filename=filename, save = False)

    if show_plot:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
