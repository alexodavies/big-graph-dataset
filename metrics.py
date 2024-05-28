import concurrent.futures
import networkx as nx
import numpy as np
from tqdm import tqdm


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