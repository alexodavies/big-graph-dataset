import os
from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from datasets.facebook_dataset import FacebookDataset
from datasets.ego_dataset import EgoDataset
from datasets.community_dataset import CommunityDataset
from datasets.cora_dataset import CoraDataset
from datasets.random_dataset import RandomDataset
from datasets.neural_dataset import NeuralDataset
from datasets.road_dataset import RoadDataset
from datasets.tree_dataset import TreeDataset
from datasets.reddit_dataset import RedditDataset
from datasets.lattice_dataset import LatticeDataset
from datasets.from_ogb_dataset import FromOGBDataset





def get_chemical_datasets(transforms, num, stage="train"):
    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")
    print(f"stage: {stage}")

    if stage == "train" or stage == "train-adgcl":
        names = ["ogbg-molpcba"]
    else:
        print("Not train stage")
        names = ["ogbg-molesol", "ogbg-molclintox",
                 "ogbg-molfreesolv", "ogbg-mollipo", "ogbg-molhiv",
                "ogbg-molbbbp", "ogbg-molbace",
                 ]

    print(f"Molecular datasets: {names}")
    datasets = [PygGraphPropPredDataset(name=name, root='./original_datasets/', transform=transforms) for name in names]
    print(datasets)
    split_idx = [data.get_idx_split() for data in datasets]

    if stage == "val":
        train_datasets = [data[split_idx[i]["train"]] for i, data in enumerate(datasets)]
        val_datasets = [data[split_idx[i]["valid"]] for i, data in enumerate(datasets)]

    elif stage == "train-adgcl":
        datasets = [data[split_idx[i]["train"]] for i, data in enumerate(datasets)]

    else:
        datasets = [data[split_idx[i][stage]] for i, data in enumerate(datasets)]

    # Need to convert to pyg inmemorydataset
    num = num if stage != "train" else 5*num
    if stage != "val":
        datasets = [FromOGBDataset(os.getcwd()+'/original_datasets/'+names[i], data, num=num, stage = stage)
                    for i, data in enumerate(datasets)] #  if names[i] != "ogbg-molpcba" else 5*num, stage=stage
    else:
        # Include train data in validation for fine tuning if dataset is evaluation only (ie not molpcba)
        datasets = [FromOGBDataset(os.getcwd()+'/original_datasets/'+names[i], data, num=num, stage = "train") + FromOGBDataset(os.getcwd()+'/original_datasets/'+names[i], val_datasets[i], num=num, stage = "val")
                    for i, data in enumerate(train_datasets)] #  if names[i] != "ogbg-molpcba" else 5*num, stage=stage

    return datasets, names

def get_social_datasets(transforms, num, stage = "train", exclude = None):
    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")

    if stage == "train":
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', num=num, stage = stage)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', num=num, stage=stage)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', num=num, stage = stage)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', num=num, stage=stage)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', num=num, stage=stage)),
            transforms(RedditDataset(os.getcwd() + '/original_datasets/' + 'reddit', num=num, stage=stage))]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "reddit"]

        if exclude is None:
            print(f"Not excluding any social datasets")
            pass

        elif isinstance(exclude, list):
            print(f"Excluding {exclude} as list")
            out_data, out_names = [], []

            for iname, name in enumerate(names):
                if name not in exclude:
                    out_data.append(social_datasets[iname])
                    out_names.append(name)

        elif isinstance(exclude, str):
            print(f"Excluding {exclude} as string")
            out_data, out_names = [], []
            for iname, name in enumerate(names):
                if name not in [exclude]:
                    out_data.append(social_datasets[iname])
                    out_names.append(name)
                else:
                    print("Passing dataset")

            social_datasets = out_data
            names = out_names

    elif stage == "val":
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', stage=stage, num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', stage=stage, num=num)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', stage=stage, num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num)),
            transforms(RedditDataset(os.getcwd() + '/original_datasets/' + 'reddit', num=num, stage=stage)),
            transforms(TreeDataset(os.getcwd() + '/original_datasets/' + 'trees', stage=stage, num=num)),
            transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=num)),
            transforms(CommunityDataset(os.getcwd() + '/original_datasets/' + 'community', stage=stage, num=num))
            ]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "reddit", "trees", "random", "community"]
    else:
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', stage=stage, num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', stage=stage, num=num)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', stage=stage, num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num)),
            transforms(RedditDataset(os.getcwd() + '/original_datasets/' + 'reddit', num=num, stage=stage)),
            transforms(TreeDataset(os.getcwd() + '/original_datasets/' + 'trees', stage=stage, num=num)),
            transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=num)),
            transforms(CommunityDataset(os.getcwd() + '/original_datasets/' + 'community', stage=stage, num=num))
            ]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "reddit", "trees", "random", "community"]

    return social_datasets, names

def get_test_datasets(transforms, num=2000, mol_only=False):
    """
    Get the test split of each dataset.

    Args:
        transforms (list): List of data transformations to apply.
        num (int): Number of samples in non-molecule datasets to include (default is 2000).
        mol_only (bool): Flag indicating whether to include only chemical datasets (default is False).

    Returns:
        tuple: A tuple containing two elements:
            - datasets (list): List of test datasets.
            - names (list): List of dataset names.
    """

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, -1, stage="test")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="test")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names


def get_val_datasets(transforms, num=2000, mol_only=False):
    """
    Get validation splits for each dataset.

    Args:
        transforms (list): List of data transformations to apply.
        num (int, optional): Number of samples in non-molecule datasets to include. Defaults to 2000.
        mol_only (bool, optional): Flag indicating whether to include only chemical datasets. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            - datasets (list): List of validation datasets.
            - names (list): List of dataset names.
    """
    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, -1, stage="val")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="val")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

def get_train_datasets(transforms, num=2000, mol_only=False):
    """
    Get the training splits of each dataset.

    Args:
        transforms (list): List of data transformations to apply.
        num (int): Number of datasets to retrieve.
        mol_only (bool): Flag indicating whether to retrieve only chemical datasets.

    Returns:
        tuple: A tuple containing the datasets and their names.
    """

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="train")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="train")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

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
    
    # All the train datasets
    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="train")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="train")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    val_chemical_datasets, val_ogbg_names = get_chemical_datasets(transforms, -1, stage="val")
    if not mol_only:
        val_social_datasets, val_social_names = get_social_datasets(transforms, num, stage="val")
    else:
        val_social_datasets = []
        val_social_names = []

    datasets = datasets + val_chemical_datasets + val_social_datasets
    all_names = ogbg_names + social_names + val_ogbg_names + val_social_names

    remove_duplicates_in_place(datasets, all_names)

    return datasets, all_names

