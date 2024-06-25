import numpy as np
import networkx as nx
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import erdos_renyi_graph

def vis_from_pyg(data, filename = None):
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    # dropped_nodes = np.ones(labels.shape[0]).astype(bool)
    for ilabel in range(labels.shape[0]):
        if ilabel not in np.unique(edges):
            g.add_node(ilabel)
    # labels = labels[dropped_nodes]

    fig, ax = plt.subplots(figsize = (6,6))

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    nx.draw_networkx_nodes(g, pos = pos, node_color=labels, cmap="tab20",
                           vmin = 0, vmax = 20, ax = ax)

    ax.axis('off')

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def get_random_graph():

    size = np.random.randint(low = 12, high = 128)

    rho = 0.05 + 0.25 * np.random.random()

    edge_index = erdos_renyi_graph(size, rho)

    G = Data(edge_index)

    return G, rho

def get_random_dataset(num = 1000):

    nx_graph_list_rhos = [get_random_graph() for _ in tqdm(range(num), leave=False)]
    datalist = [item[0] for item in nx_graph_list_rhos]
    rhos= [item[1] for item in nx_graph_list_rhos]

    for idata, data in enumerate(datalist):
        data.y = torch.Tensor([rhos[idata]])
        datalist[idata] = data


    return datalist





class RandomDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`

    Dataset of random erdos-renyi graphs, between 12 and 128 nodes, of random density between 0.05 and 0.3.

    The target is the density of each graph.

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
            print("Random files exist")
            return

        data_list = get_random_dataset(num=self.num)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # if self.stage != "train":
        #     for i, data in enumerate(data_list):
        #         vis_from_pyg(data, filename=self.root + f'/processed/{self.stage}-{i}.png')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])