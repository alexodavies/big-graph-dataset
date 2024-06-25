import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from utils import describe_one_dataset

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred import PygGraphPropPredDataset


full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()



def to_onehot_atoms(x):
    one_hot_tensors = []
    for i, num_values in enumerate(full_atom_feature_dims):
        one_hot = torch.nn.functional.one_hot(x[:, i], num_classes=num_values)
        one_hot_tensors.append(one_hot)

    return torch.cat(one_hot_tensors, dim=1)

def to_onehot_bonds(x):
    one_hot_tensors = []
    for i, num_values in enumerate(full_bond_feature_dims):
        one_hot = torch.nn.functional.one_hot(x[:, i], num_classes=num_values)
        one_hot_tensors.append(one_hot)

    return torch.cat(one_hot_tensors, dim=1)

class FromOGBDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`

    Converts an Open Graph Benchmark dataset into a `torch_geometric.data.InMemoryDataset`.
    This allows standard dataset operations like concatenation with other datasets.

    The Open Graph Benchmark project is available here:

         `Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." Advances in neural information processing systems 33 (2020): 22118-22133.`

    We convert atom and bond features into one-hot encodings.
    The resulting shapes are:
     - node (atom features): (174, N Atoms)
     - edge (bond features) features: (13, N Bonds)

    Args:
        root (str): Root directory where the dataset should be saved.
        ogb_dataset (list): an `PygGraphPropPredDataset` to be converted back to `InMemoryDataset`.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. -1 takes all available samples for that stage. (default: :obj:`-1`).
    """
    def __init__(self, root, ogb_dataset, stage = "train", num = -1, transform=None, pre_transform=None, pre_filter=None):
        self.ogb_dataset = ogb_dataset
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2,
                               "train-adgcl":3}
        self.num = num
        print(f"Converting OGB stage {self.stage}")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['dummy.csv']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.
        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print(f"\nTU files exist at {self.processed_paths[self.stage_to_index[self.stage]]}")
            return
        data_list = self.ogb_dataset

        num_samples = len(data_list)
        if num_samples < self.num:
            keep_n = num_samples
        else:
            keep_n = self.num

        new_data_list = []
        for i, item in enumerate(data_list[:keep_n]):


            data = Data(x = to_onehot_atoms(item.x), 
                        edge_index=item.edge_index,
                        edge_attr= to_onehot_bonds(item.edge_attr), 
                        y = item.y)
            
            new_data_list.append(data)
        data_list = new_data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


def from_ogb_dataset(root,  stage="train", num=-1):
    """
    Load a dataset from the Open Graph Benchmark (OGB) and convert it to the Big Graph Dataset format.

    Args:
        name (str): The name of the OGB dataset. (Classification: "ogbg-molpcba", "ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast") (Regression: "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo")
        stage (str, optional): The stage of the dataset to load (e.g., "train", "valid", "test"). Defaults to "train".
        num (int, optional): The number of samples to load. Set to -1 to load all samples. Defaults to -1.

    Returns:
        FromOGBDataset: The converted dataset in the Big Graph Dataset format.
    """
    dataset = PygGraphPropPredDataset(root.split('/')[-1], root = root)
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx["valid" if stage == "val" else stage]]
    return FromOGBDataset(root, dataset, stage = stage, num = num)

if __name__ == "__main__":
    molpcba = from_ogb_dataset("ogbg-molpcba", stage = "train")
    describe_one_dataset(molpcba)