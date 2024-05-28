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
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', num=num, stage=stage))]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly"]

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
            transforms(TreeDataset(os.getcwd() + '/original_datasets/' + 'trees', stage=stage, num=num)),
            transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=num)),
            transforms(CommunityDataset(os.getcwd() + '/original_datasets/' + 'community', stage=stage, num=num))
            ]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "trees", "random", "community"]
    else:
        social_datasets = [
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', stage=stage, num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', stage=stage, num=num)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', stage=stage, num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num)),
            transforms(TreeDataset(os.getcwd() + '/original_datasets/' + 'trees', stage=stage, num=num)),
            transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=num)),
            transforms(CommunityDataset(os.getcwd() + '/original_datasets/' + 'community', stage=stage, num=num))
            ]
        names = ["facebook_large", "twitch_egos", "cora", "roads", "fruit_fly", "trees", "random", "community"]

    return social_datasets, names

def get_test_datasets(transforms, num = 2000, mol_only = False):

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="test")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="test")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names


def get_val_datasets(transforms, num = 2000, mol_only = False):
    print("Getting val datasets")
    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="val")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="val")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

def get_train_datasets(transforms, num = 2000, mol_only = False):
    
    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="train")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="train")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

