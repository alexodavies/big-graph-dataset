import os
import networkx as nx
from networkx.algorithms.planarity import is_planar
import pandas as pd
import torch
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset, Data
import pickle
import wget
import numpy as np
import gzip
import shutil
from utils import ESWR, describe_one_dataset
from littleballoffur.exploration_sampling import *

print(os.getcwd())

def four_cycles(g):
    """
    Returns the number of 4-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, 4)
    return len(list(cycles)) / g.order()


def download_roads(visualise = False):
    zip_url = "https://snap.stanford.edu/data/roadNet-PA.txt.gz"

    start_dir = os.getcwd()
    os.chdir("bgd_files")


    if "roads" not in os.listdir():
        print("Downloading roads graph")
        os.mkdir("roads")
        os.chdir("roads")

        _ = wget.download(zip_url)

        with gzip.open('roadNet-PA.txt.gz', 'rb') as f_in:
            with open('roadNet-PA.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove("roadNet-PA.txt.gz")
    else:
        os.chdir("roads")

        if "road-graph.npz" in os.listdir():
            with open("road-graph.npz", "rb") as f:
                graph = pickle.load(f)
            os.chdir('../')
            return graph


    edgelist = pd.read_csv("roadNet-PA.txt", delimiter = "\t", header=3)

    graph = nx.Graph()

    sources = edgelist["# FromNodeId"].to_numpy().astype("int")
    targets = edgelist["ToNodeId"].to_numpy().astype("int")

    edges_to_add = [(sources[i], targets[i]) for i in tqdm(range(sources.shape[0]))]

    graph.add_edges_from(edges_to_add)

    del edges_to_add


    graph = nx.convert_node_labels_to_integers(graph)

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    with open("road_graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)
    return graph


def individual_sample(graph):
    sampler = DiffusionSampler(number_of_nodes=np.random.randint(12, 48))
    return nx.convert_node_labels_to_integers(sampler.sample(graph))

def add_attrs_to_graph(g):
    dummy_label = torch.Tensor([1])
    nx.set_edge_attributes(g, dummy_label, "attrs")
    nx.set_node_attributes(g, dummy_label, "attrs")


    return g


def get_road_dataset(num = 2000, targets = False):
    fb_graph = download_roads()
    nx_graph_list = ESWR(fb_graph, num, 96)


    data_objects = [pyg.utils.from_networkx(g) for g in nx_graph_list]

    for i, data in enumerate(tqdm(data_objects, desc="Calculating diameter values for roads", leave=False)):
        data.y = torch.tensor(int(is_planar(nx_graph_list[i])))

        nx_graph_list[i] = data

    return data_objects# loader

class RoadDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    

    Road graphs from Pennsylvania, sampled from a large original graph using ESWR.
    The original graph is sourced from:

         `J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics 6(1) 29--123, 2009.`

    The task is predicting whether a given graph is planar (can be laid out with no crossing edges).

     - Task: Graph classification
     - Num node features: None
     - Num edge features: None
     - Num target values: 1
     - Target shape: 1
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
        _ = download_roads()
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
            print("Road files exist")
            return

        data_list = get_road_dataset(num=self.num, targets=self.stage != "train")

        # if self.stage == "train":
        #     print("Found stage train, dropping targets")
        #     new_data_list = []
        #     for i, item in enumerate(data_list):
        #         n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

        #         data = Data(x = torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
        #                     edge_index=item.edge_index,
        #                     edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
        #                     y = None)
        #         new_data_list.append(data)
        #     data_list = new_data_list
        # else:
        #     new_data_list = []
        #     for i, item in enumerate(data_list):
        #         n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


        #         data = Data(x = item.x,
        #                     edge_index=item.edge_index,
        #                     edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
        #                     y = item.y)

        #         new_data_list.append(data)
        #     data_list = new_data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])

        del data_list

if __name__ == "__main__":
    dataset = RoadDataset(os.getcwd()+'/bgd_files/'+'roads', stage = "train")
    describe_one_dataset(dataset)