import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
import wget
from tqdm import tqdm
from utils import describe_one_dataset
import zipfile

def get_twitch(num = 49152, include_targets = False):
    print("\nProcessing twitch egos dataset")
    zip_url = "https://snap.stanford.edu/data/twitch_egos.zip"
    start_dir = os.getcwd()
    os.chdir("bgd_files")

    if "twitch_edges.json" not in os.listdir("twitch_egos"):
        print("Downloading Twitch Egos")
        _ = wget.download(zip_url)
        with zipfile.ZipFile("twitch_egos.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("twitch_egos.zip")
    os.chdir("twitch_egos")


    with open("twitch_edges.json", "r") as f:
        all_edges = json.load(f)

    twitch_targets = pd.read_csv("twitch_target.csv")
    ids, targets = twitch_targets["id"], twitch_targets["target"]
    id_to_target = {ids[i]:targets[i] for i in range(len(ids))}


    graph_ids = list(all_edges.keys())

    graphs = []

    print("Entering ego processing loop")
    for id in tqdm(graph_ids[:num], leave = False):
        edges = all_edges[id]

        g = nx.Graph()

        nodes = np.unique(edges).tolist()

        for node in nodes:
            g.add_node(node) #, attr = torch.Tensor([1]))

        for edge in edges:
            g.add_edge(edge[0], edge[1]) #, attr=torch.Tensor([1]))
        graphs.append(g)

    os.chdir(start_dir)
    # data_objects = [pyg.utils.from_networkx(g, group_node_attrs=all, group_edge_attrs=all) for g in graphs]
    data_objects = [pyg.utils.from_networkx(g) for g in graphs]

    for id, data in enumerate(data_objects):
        print(id_to_target[id])
        data.y = torch.Tensor([id_to_target[id]])
        print(data.y)

    return  data_objects

class EgoDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    
    Ego networks from the streaming platform Twitch.
    The original graph is sourced from:

         `B. Rozemberczki, O. Kiss, R. Sarkar: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs 2019.`

    The task is predicting whether a given streamer plays multiple different games.

     - Task: Graph classification
     - Num node features: None
     - Num edge features: None
     - Num target values: 1
     - Target shape: 1
     - Num graphs: 127094

    Args:
        root (str): Root directory where the dataset should be saved.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. (default: :obj:`2000`).
    """

    def __init__(self, root, stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 5000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}


        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['twitch_edges.json',
                'twitch_target.json']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.
        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Ego files exist")
            return
        data_list = get_twitch(num=self.num, include_targets =self.stage != "train")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


if __name__ == "__main__":
    dataset = EgoDataset(os.getcwd()+'/bgd_files/'+'twitch_egos', stage = "train")
    describe_one_dataset(dataset)
