import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import os
from torch_geometric.data import InMemoryDataset

def get_community_graph(size = 48, proportions = [0.25, 0.25, 0.25, 0.25], P_intra = 0.5, P_inter=0.05 + 0.1*np.random.random()):

    sizes = (np.array(proportions) * size).astype(int).tolist()#


    subgraphs = []
    counter = 0
    for i_size, size in enumerate(sizes):
        g = nx.Graph()

        for i in range(counter, counter + size):
            g.add_node(i)

        counter += size
        subgraphs.append(g)

    for g in subgraphs:
        for n1 in g.nodes():
            for n2 in g.nodes():
                if np.random.random() <= P_intra:
                    g.add_edge(n1, n2)

    node_identifiers = [list(g.nodes()) for g in subgraphs]

    G = nx.Graph()
    for g in subgraphs:
        G = nx.compose(G, g)

    for ids_1 in node_identifiers:
        for ids_2 in node_identifiers:
            if ids_1 == ids_2:
                pass
            else:
                for n1 in ids_1:
                    for n2 in ids_2:
                        if np.random.random() <= P_inter:
                            G.add_edge(n1, n2)

    return G, P_inter


def get_community_dataset(num = 1000):
    nx_graph_list_rhos = [get_community_graph(P_inter=0.05 + 0.05*np.random.random()) for _ in tqdm(range(num), leave=False)]
    nx_graph_list = [item[0] for item in nx_graph_list_rhos]
    rhos= [item[1] for item in nx_graph_list_rhos]

    datalist = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in tqdm(nx_graph_list)]

    for idata, data in enumerate(datalist):
        data.y = torch.Tensor([rhos[idata]])
        datalist[idata] = data

    return datalist

class CommunityDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`

    Dataset of random community-like graphs, composed of four evenly sized subgraphs with a random inter-subgraph edge probability.
    Subgraphs have a density of 0.5, and inter-subgraph edges have a random density between 0.05 and 0.15.

    The target is the inter-subgraph density of each graph.

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
            print("Facebook files exist")
            return

        data_list = get_community_dataset(num=self.num, keep_target=self.stage != "train")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])