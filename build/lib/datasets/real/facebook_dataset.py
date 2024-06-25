import os
import networkx as nx
import pandas as pd
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import InMemoryDataset, Data
import pickle
import zipfile
import wget
import numpy as np
from utils import ESWR
import json

from littleballoffur.exploration_sampling import *

from utils import describe_one_dataset, vis_grid


def download_facebook(visualise = False):
    zip_url = "https://snap.stanford.edu/data/facebook_large.zip"

    start_dir = os.getcwd()
    os.chdir("bgd_files")

    if "facebook-graph.npz" in os.listdir():
        with open("facebook-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph

    if "musae_facebook_edges.csv" not in os.listdir("facebook_large"):
        print("Downloading FB graph")
        _ = wget.download(zip_url)
        with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("facebook_large.zip")
    os.chdir("facebook_large")

    edgelist = pd.read_csv("musae_facebook_edges.csv")
    labels = pd.read_csv("musae_facebook_target.csv")

    # Not using node features as they are all different lengths :(
    with open("musae_facebook_features.json", 'r', encoding='utf-8') as file:
            features = json.load(file)


    conversion_dict = {"company":       torch.Tensor([0]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "government":    torch.Tensor([1]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "politician":    torch.Tensor([2]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "tvshow":        torch.Tensor([3])} #, 0, 0, 0, 0, 0, 0, 0, 0]),


    graph = nx.Graph()
    label_specific = labels["page_type"]
    for col in labels["id"]:
        graph.add_node(int(col))
        # Turns out the facebook data has attributes of varying length
        graph.nodes[int(col)]["label"] = conversion_dict[label_specific[col]]
        # graph.nodes[int(col)]["attrs"] = torch.Tensor(features[str(col)])

    sources = edgelist["id_1"].to_numpy().astype("int")
    targets = edgelist["id_2"].to_numpy().astype("int")

    for i in range(sources.shape[0]):
        graph.add_edge(sources[i], targets[i])

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    os.chdir(start_dir)
    return graph

def specific_from_networkx(graph):
    node_labels = []
    edge_indices = []
    # node_attrs = []

    # Collect node labels and attributes
    for n in list(graph.nodes(data=True)):
        node_labels.append(n[1]["label"])
        # Weird features, different sizes, so not including
        # node_attrs.append(n[1]["attrs"])

    # Collect edge indices and attributes
    for e in graph.edges(data=True):
        edge_indices.append((e[0], e[1]))

        # uncomment for edge attributes:
        # edge_attrs.append(e[2]["attr"]) 

    # Convert to PyTorch tensors
    node_labels = torch.stack(node_labels).flatten()

    # Specific to classification on nodes! Hard coding num classes as this happens on a per-graph basis
    node_labels = one_hot(node_labels.to(int), num_classes = 4)

    # node_attrs = torch.stack(node_attrs)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    data = Data(x=None, edge_index=edge_indices, edge_attr = None,  y=node_labels)

    return data

def get_fb_dataset(num = 2000, targets = False):
    fb_graph = download_facebook()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 96)
    data_objects = [specific_from_networkx(g) for g in nx_graph_list]

    return data_objects# loader

class FacebookDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    

    Facebook page-to-page interaction graphs, sampled from a large original graph using ESWR.
    The original graph is sourced from:

         `Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-Scale Attributed Node Embedding.  Journal of Complex Networks 2021`

    The original data has node features, but as they are of varying length, we don't include them here.

    The task is node classification for the category of each Facebook page in a given graph, one-hot encoded for four categories.

     - Task: Node classification
     - Num node features: None
     - Num edge features: None
     - Num target values: 4
     - Target shape: N Nodes
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
        
        self.task = "node-classification"

        # _ = download_facebook()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['musae_facebook_edges.csv',
                'musae_facebook_target.csv']


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

        data_list = get_fb_dataset(num=self.num, targets=self.stage != "train")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


if __name__ == "__main__":
    dataset = FacebookDataset(os.getcwd()+'/bgd_files/'+'facebook_large', stage = "train", num = 5000)
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/facebook_large/train.png")
    
    dataset = FacebookDataset(os.getcwd()+'/bgd_files/'+'facebook_large', stage = "val", num = 1000)
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/facebook_large/val.png")
    
    dataset = FacebookDataset(os.getcwd()+'/bgd_files/'+'facebook_large', stage = "test", num = 1000)
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/facebook_large/test.png")