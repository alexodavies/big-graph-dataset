import os
import networkx as nx
import pickle
import pandas as pd
import zipfile
import numpy as np
import gzip
import shutil
import torch
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.io import read_npz
from torch.nn.functional import one_hot
import wget
import matplotlib.pyplot as plt
from ..utils import describe_one_dataset, vis_grid, ESWR
from ..utils import ESWR, CustomForestFireSampler

def download_livejournal():
    # This function should download the data and process it into one big networkx graph

    # If node labels are the target, they should be included as an attribute of each node
    # In this case each node is (id, {"attrs":..., "label":...})
    # Probably the same is best for edge labels

    # url to data
    zip_url = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
    start_dir = os.getcwd()
    os.chdir("bgd_files")

    # create directory if we haven't before
    if "livejournal" not in os.listdir():
        print("Downloading LIVEJOURNAL")
        os.mkdir("livejournal")
        os.chdir("livejournal")
    # Means that we've previously downloaded data
    else:
        os.chdir("livejournal")

    if "livejournal-graph.npz" in os.listdir():
        with open("livejournal-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph

    if "soc-LiveJournal1.txt" not in os.listdir():
        print(f"Downloading livejournal graph from {zip_url}")
        _ = wget.download(zip_url)
        with gzip.open("soc-LiveJournal1.txt.gz", 'rb') as zip_ref:
            with open('soc-LiveJournal1.txt', 'wb') as f:
                shutil.copyfileobj(zip_ref, f)
        os.remove("soc-LiveJournal1.txt.gz")

    edgelist = pd.read_csv("soc-LiveJournal1.txt", delimiter = "\t", header=3)

    # graph = nx.Graph()
    print("Obtained livejournal edgelist")
    sources = edgelist["# FromNodeId"].to_numpy().astype("int")[:100000]
    targets = edgelist["ToNodeId"].to_numpy().astype("int")[:100000]
    print("Adding livejournal edges")
    edges_to_add = [(sources[i], targets[i]) for i in tqdm(range(sources.shape[0]), leave = False, colour="GREEN")]

    # Dictionary approach
    edge_dict = {}
    for u, v in edges_to_add:
        edge_dict.setdefault(u, []).append(v)
        edge_dict.setdefault(v, []).append(u)
    graph = nx.Graph(edge_dict)
    # graph.add_edges_from(edges_to_add)

    # graph = nx.from_edgelist(edges_to_add, create_using=nx.Graph)

    del edges_to_add

    print("Converting livejournal labels to integers")
    graph = nx.convert_node_labels_to_integers(graph)

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    print("Saving livejournal graph")
    # sampler = MettFireSampler(number_of_nodes=50000)
    # graph = sampler.sample(graph)
    with open("livejournal-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)
    return graph

def get_livejournal_dataset(num = 2000):
    graph = download_livejournal()
    nx_graph_list = ESWR(graph, num, 96)

    data_objects = [pyg.utils.from_networkx(g) for g in nx_graph_list]

    return data_objects# loader

class LivejournalDataset(InMemoryDataset):
    # Documentation is essential! Without the sources listed I won't be able to include the dataset
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    
    LiveJournal is a free on-line community with almost 10 million members; a significant fraction of these members are highly active. 
    (For example, roughly 300,000 update their content in any given 24-hour period.) 
    LiveJournal allows members to maintain journals, individual and group blogs, and it allows people to declare which other members are their friends they belong.

    The original graph is sourced from:

         `L. Backstrom, D. Huttenlocher, J. Kleinberg, X. Lan. Group Formation in Large Social Networks: Membership, Growth, and Evolution. KDD, 2006.`

    There are no node or edge features.

    There is also no set task, although edge prediction is a valid option.


    Args:
        root (str): Root directory where the dataset should be saved.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. (default: :obj:`2000`).
    """
    def __init__(self, root, stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}

        # Options are node-classification, node-regression, graph-classification, graph-regression, edge-regression, edge-classification
        # Graph-level tasks are preferred! (graph-classification and graph-regression)
        # edge-prediction is another option if you can't think of a good task
        self.task = "edge-prediction"

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])


    @property
    def raw_file_names(self):
        # Replace with your saved raw file name
        return ['livejournal-graph.npz']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print(f"Livejournal files exist")
            return

        # Get a list of num pytorch_geometric.data.Data objects
        data_list = get_livejournal_dataset(num=self.num)

        # You can iterate over the data objects if necessary:
        # ===================================================
        # new_data_list = []
        # for i, item in enumerate(data_list):
        #     n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

        #     data = Data(x = item.x,
        #                 edge_index=item.edge_index,
        #                 # Here we don't have any edge features so we use just 1s
        #                 edge_attr=None,
        #                 y = item.y)

        #     new_data_list.append(data)
        # data_list = new_data_list
        # ===================================================

        # Torch geometric stuff
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


# Run with python -m bgd/example_dataset.py
# Run this to check that everything works!
if __name__ == "__main__":
    # Please set the last part of the path to your dataset name!
    dataset = LivejournalDataset(os.getcwd()+'/bgd_files/'+'livejournal', stage = "train")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/livejournal/train.png")

    dataset = LivejournalDataset(os.getcwd()+'/bgd_files/'+'livejournal', stage = "val")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/livejournal/val.png")

    dataset = LivejournalDataset(os.getcwd()+'/bgd_files/'+'livejournal', stage = "test")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/livejournal/test.png")