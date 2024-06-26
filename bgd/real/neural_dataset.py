import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import os

from utils import describe_one_dataset, vis_grid

from torch_geometric.data import InMemoryDataset, Data
import inspect
from utils import ESWR
from littleballoffur.exploration_sampling import *
import littleballoffur.exploration_sampling as samplers
import sys
import pickle
import zipfile
import wget

def four_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, length_bound=4)
    return len(list(cycles))

def load_fly(return_tensor = False):
    start_dir = os.getcwd()
    os.chdir("fruit_fly")
    os.chdir("Supplementary-Data-S1")

    data_path = os.path.join(os.getcwd(), "all-all_connectivity_matrix.csv")
    fly_mat = pd.read_csv(
        data_path).drop(
        columns=['Unnamed: 0'])
    fly_mat = fly_mat.to_numpy()

    os.chdir(start_dir)
    os.chdir("bgd_files")

    if "fruit_fly" not in os.listdir():
        os.mkdir("fruit_fly")
        os.chdir("fruit_fly")

    # Could be fun to trim only to multiple-synapse connections?
    # fly_mat[fly_mat <= 2] = 0
    # fly_mat[fly_mat > 2] = 1
    fly_mat[np.identity(fly_mat.shape[0], dtype=bool)] = 0.
    fly_graph = fly_mat

    nx_graph = nx.from_numpy_array(fly_graph, create_using=nx.Graph)

    CGs = [nx_graph.subgraph(c) for c in nx.connected_components(nx_graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    nx_graph = CGs[0]
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    os.chdir(start_dir)

    return nx_graph

def specific_from_networkx(graph):
    # Turns a graph into a pytorch geometric object
    # Mostly by unpacking dictionaries on nodes and edges
    # Here edge labels are the target
    edge_labels = []
    edge_indices = []

    # Collect edge indices and attributes
    for e in graph.edges(data=True):
        # graph.edges(data=True) is a generator producing (node_id1, node_id2, {attribute dictionary})
        edge_indices.append((e[0], e[1]))
        edge_labels.append(torch.Tensor([e[2]["weight"]]))


    # Specific to classification on edges! This is a binary edge classification (pos/neg) task
    edge_labels = torch.Tensor(edge_labels).reshape(-1,1)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    # Can pass:
    # x:            node features, shape (n nodes x n features)
    # edge_index:   the list of edges in the graph, shape (2, n_edges). Entries edge_index[i, :] are [node_id1, node_id2].
    # edge_attr:    edge features, shape (n_edges, n_features), same order as edgelist
    # y:            targets. Graph regression shape (n_variables), graph classification (n_classes), node classification (n_nodes, n_classes), edge classification (n_edges, n_classes)
    data = Data(x=None, edge_index=edge_indices, edge_attr = None,  y=edge_labels)

    return data

def get_fly_dataset(num = 2000):
    fb_graph = load_fly()
    nx_graph_list = ESWR(fb_graph, num, 96)

    datalist = [specific_from_networkx(item) for item in nx_graph_list]

    return datalist

class NeuralDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    
    A dataset of the connectome of a fruit fly larvae.
    The original graph is sourced from:

         `Michael Winding et al. , The connectome of an insect brain.Science379,eadd9330(2023).DOI:10.1126/science.add9330`

    We process the original multigraph into ESWR samples of this neural network, with predicting the strength of the connection (number of synapses) between two neurons as the target.

     - Task: Edge regression
     - Num node features: 0
     - Num edge features: 0
     - Num target values: 1
     - Target shape: N Edges
     - Num graphs: Parameterised by `num`

    Args:
        root (str): Root directory where the dataset should be saved.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. (default: :obj:`2000`).
    """

    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        
        self.task = "edge-regression"
        _ = load_fly()
        del _
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return []


    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Connectome files exist")
            return

        data_list = get_fly_dataset(num=self.num)

        # if self.stage == "train":
        #     print("Found stage train, dropping targets")
            # new_data_list = []
            # for i, item in enumerate(data_list):
            #     n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

            #     data = Data(x = torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
            #                 edge_index=item.edge_index,
            #                 edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
            #                 y = None)

            #     new_data_list.append(data)
            # data_list = new_data_list
        # else:
        # new_data_list = []
        # for i, item in enumerate(data_list):
        #     n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


        #     data = Data(x = None,
        #                 edge_index=item.edge_index,
        #                 edge_attr=None,
        #                 y = item.y)
        #     new_data_list.append(data)
        # data_list = new_data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # if self.stage != "train":
        #     for i, data in enumerate(data_list):
        #         vis_from_pyg(data, filename=self.root + f'/processed/{self.stage}-{i}.png')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])

        del data_list


if __name__ == "__main__":
    dataset = NeuralDataset(os.getcwd()+'/bgd_files/'+'fruit_fly', stage = "train")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/fruit_fly/train.png")
    
    dataset = NeuralDataset(os.getcwd()+'/bgd_files/'+'fruit_fly', stage = "val")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/fruit_fly/val.png")
    
    dataset = NeuralDataset(os.getcwd()+'/bgd_files/'+'fruit_fly', stage = "test")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/fruit_fly/test.png")