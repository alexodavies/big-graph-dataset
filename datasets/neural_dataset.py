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
    pwd = os.getcwd()
    zip_url = "https://www.kaggle.com/datasets/alexanderowendavies/fruit-fly-larva-brain/download" #?datasetVersionNumber=1"

    start_dir = os.getcwd()
    os.chdir("fruit_fly")
    os.chdir("Supplementary-Data-S1")



    data_path = os.path.join(os.getcwd(), "all-all_connectivity_matrix.csv")
    fly_mat = pd.read_csv(
        data_path).drop(
        columns=['Unnamed: 0'])
    fly_mat = fly_mat.to_numpy()
    #

    os.chdir(start_dir)
    os.chdir("original_datasets")

    if "fruit_fly" not in os.listdir():
        os.mkdir("fruit_fly")
        os.chdir("fruit_fly")

    # Could be fun to trim only to multiple-synapse connections?

    fly_mat[fly_mat <= 2] = 0
    fly_mat[fly_mat > 2] = 1
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

def add_attrs_to_graph(g):
    dummy_label = torch.Tensor([1])
    nx.set_edge_attributes(g, dummy_label, "attrs")
    nx.set_node_attributes(g, dummy_label, "attrs")

    return g


def get_fly_dataset(num = 2000, targets = False):
    fb_graph = load_fly()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 96)
    nx_graph_list = [add_attrs_to_graph(g) for g in nx_graph_list]

    datalist = [pyg.utils.from_networkx(item, group_edge_attrs=all, group_node_attrs=all) for item in nx_graph_list]

    for i, data in enumerate(datalist):
        if targets:
            data.y = torch.tensor(four_cycles(nx_graph_list[i]) * 0.01, dtype=float)
        else:
            data.y = None 
        data.edge_attr = data.edge_attr[:,0].reshape(-1,1)
        datalist[i] = data


    return datalist# loader

class NeuralDataset(InMemoryDataset):
    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
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

        data_list = get_fly_dataset(num=self.num, targets=self.stage != "train")

        if self.stage == "train":
            print("Found stage train, dropping targets")
            new_data_list = []
            for i, item in enumerate(data_list):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

                data = Data(x = torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
                            edge_index=item.edge_index,
                            edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                            y = None)

                new_data_list.append(data)
            data_list = new_data_list
        else:
            new_data_list = []
            for i, item in enumerate(data_list):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


                data = Data(x = item.x,
                            edge_index=item.edge_index,
                            edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                            y = item.y)
                new_data_list.append(data)
            data_list = new_data_list

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
    dataset = NeuralDataset(os.getcwd()+'/original_datasets/'+'fruit_fly', stage = "train")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/original_datasets/fruit_fly/train.png")
    
    dataset = NeuralDataset(os.getcwd()+'/original_datasets/'+'fruit_fly', stage = "val")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/original_datasets/fruit_fly/val.png")
    
    dataset = NeuralDataset(os.getcwd()+'/original_datasets/'+'fruit_fly', stage = "test")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/original_datasets/fruit_fly/test.png")