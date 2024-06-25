import numpy as np
import networkx as nx
import torch
import os
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset



def get_tree_graph(max_nodes = 96):



    # base_tensor = torch.Tensor([np.random.randint(1,9)])

    n = np.random.randint(low=8, high = max_nodes)
    G = nx.random_tree(n = n)
    # G_attr = nx.Graph()

    # for i in range(G.order()):
    #     G_attr.add_node(i)

    # for (n1, n2) in G.edges():
    # # for n1 in G.nodes():
    # #     for n2 in G.nodes():
    #     G_attr.add_edge(n1, n2)

    depth = nx.eccentricity(G, 0) / G.order()
    return G, depth


def get_tree_dataset(num = 1000):
    nx_graph_list_rhos = [get_tree_graph() for _ in tqdm(range(num), leave=False)]
    nx_graph_list = [item[0] for item in nx_graph_list_rhos]
    depths= [item[1] for item in nx_graph_list_rhos]
    Ns = [graph.order() for graph in nx_graph_list]

    datalist = [pyg.utils.from_networkx(g) for g in tqdm(nx_graph_list)]

    for idata, data in enumerate(datalist):
        data.y = torch.Tensor([depths[idata]])
        datalist[idata] = data

    return datalist


class TreeDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`

    Dataset of random tree structures, between 8 and 96 nodes, produced with `networkx.random_tree`.

    The target is the depth of the tree, normalised by the number of nodes in the tree.

    Args:
        root (str): Root directory where the dataset should be saved.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. -1 takes all available samples for that stage. (default: :obj:`-1`).
    """
    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        # _ = download_facebook()
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
            print("Tree files exist")
            return

        data_list = get_tree_dataset(num=self.num)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])