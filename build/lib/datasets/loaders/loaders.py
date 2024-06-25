import os
from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from datasets.real.facebook_dataset import FacebookDataset
from datasets.real.ego_dataset import EgoDataset
from datasets.real.cora_dataset import CoraDataset
from datasets.real.neural_dataset import NeuralDataset
from datasets.real.road_dataset import RoadDataset
from datasets.real.reddit_dataset import RedditDataset
from datasets.real.from_ogb_dataset import FromOGBDataset, from_ogb_dataset
from datasets.real.from_tu_dataset import from_tu_dataset

from datasets.synthetic.random_dataset import RandomDataset
from datasets.synthetic.tree_dataset import TreeDataset
from datasets.synthetic.lattice_dataset import LatticeDataset
from datasets.synthetic.community_dataset import CommunityDataset


def get_datasets(transforms, num, stage="train", exclude=None):
    if "bgd_files" not in os.listdir():
        os.mkdir("bgd_files")
        
    all_datasets = {
        "facebook_large": FacebookDataset,
        "twitch_egos": EgoDataset,
        "cora": CoraDataset,
        "roads": RoadDataset,
        "fruit_fly": NeuralDataset,
        "reddit": RedditDataset,
        "trees": TreeDataset,
        "random": RandomDataset,
        "community": CommunityDataset
    }

    ogb_names = ["ogbg-molpcba", "ogbg-molesol", "ogbg-molclintox",
                 "ogbg-molfreesolv", "ogbg-mollipo", "ogbg-molhiv",
                 "ogbg-molbbbp", "ogbg-molbace"]
    
    all_datasets.update({name: from_ogb_dataset for name in ogb_names})

    tu_names = ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "IMDB-BINARY", "REDDIT-BINARY"]

    all_datasets.update({name: from_tu_dataset for name in tu_names})

    selected_datasets = list(all_datasets.keys())

    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        selected_datasets = [d for d in selected_datasets if d not in exclude]
    print(stage)
    datasets = [transforms(all_datasets[name](os.path.join(os.getcwd(), 'bgd_files', name), num=num, stage=stage))
                for name in selected_datasets]

    return datasets, selected_datasets

def get_test_datasets(transforms, num=2000, mol_only=False):
    """
    Get the test split of each dataset.

    Args:
        transforms (list): List of data transformations to apply.
        num (int): Number of samples in datasets to include (default is 2000).
        mol_only (bool): Flag indicating whether to include only chemical datasets (default is False).

    Returns:
        tuple: A tuple containing two elements:
            - datasets (list): List of test datasets.
            - names (list): List of dataset names.
    """

    datasets, names = get_datasets(transforms, num, stage="test")
    return datasets, names


def get_val_datasets(transforms, num=2000, mol_only=False):
    """
    Get validation splits for each dataset.

    Args:
        transforms (list): List of data transformations to apply.
        num (int, optional): Number of samples in datasets to include. Defaults to 2000.
        mol_only (bool, optional): Flag indicating whether to include only chemical datasets. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            - datasets (list): List of validation datasets.
            - names (list): List of dataset names.
    """

    datasets, names = get_datasets(transforms, num, stage="val")
    return datasets, names

def get_train_datasets(transforms, num=2000, mol_only=False):
    """
    Get the training splits of each dataset.

    Args:
        transforms (list): List of data transformations to apply.
        num (int): Number of datasets to retrieve.
        mol_only (bool): Flag indicating whether to retrieve only chemical datasets.

    Returns:
        tuple: A tuple containing two elements:
            - datasets (list): A list of all the datasets.
            - all_names (list): A list of names corresponding to each dataset.
    """

    datasets, names = get_datasets(transforms, num, stage= "train", exclude = ["community", "trees", "random"])
    return datasets, names

def remove_duplicates_in_place(list1, list2):
    seen = set()
    i = 0
    
    while i < len(list2):
        if list2[i] in seen:
            del list1[i]
            del list2[i]
        else:
            seen.add(list2[i])
            i += 1

def get_all_datasets(transforms, num=5000, mol_only=False):
    """
    Get all datasets for training and validation, in that order.

    Args:
        transforms (list): List of data transformations to apply to the datasets.
        num (int, optional): Number of samples to load from each dataset. Defaults to 5000.
        mol_only (bool, optional): Flag indicating whether to include only chemical datasets. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            - datasets (list): A list of all the datasets.
            - all_names (list): A list of names corresponding to each dataset.
    """

    train_datasets, train_names = get_train_datasets(transforms, num)
    val_datasets, val_names = get_val_datasets(transforms, -1)
    test_datasets, test_names = get_val_datasets(transforms, -1)

    datasets = train_datasets + val_datasets + test_datasets
    all_names = train_names + val_names + test_names

    remove_duplicates_in_place(datasets, all_names)

    return datasets, all_names

