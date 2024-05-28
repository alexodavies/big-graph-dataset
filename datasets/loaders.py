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
            transforms(FacebookDataset(os.getcwd() + '/original_datasets/' + 'facebook_large', num=num)),
            transforms(EgoDataset(os.getcwd() + '/original_datasets/' + 'twitch_egos', num=num, stage=stage)),
            transforms(CoraDataset(os.getcwd() + '/original_datasets/' + 'cora', num=num)),
            transforms(RoadDataset(os.getcwd() + '/original_datasets/' + 'roads', stage=stage, num=num)),
            transforms(NeuralDataset(os.getcwd() + '/original_datasets/' + 'fruit_fly', stage=stage, num=num))]
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

def get_random_datasets(transforms, num, stage = "train"):
    if "original_datasets" not in os.listdir():
        os.mkdir("original_datasets")

    if stage == "train":
        social_datasets = [transforms(RandomDataset(os.getcwd() + '/original_datasets/' + 'random', stage=stage, num=10 * num))]
        names = ["random"]
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

def get_train_loader(batch_size, transforms,
                     subset = ["chemical", "social"],
                     num_social = 50000,
                     social_excludes = None,
                     for_adgcl = False):
    """
    Prepare a torch concat dataset dataloader
    Args:
        dataset: original dataset - hangover from early code, will be removed in future version
        batch_size: batch size for dataloader
        transforms: transforms applied to each dataset
        num_social: number of graphs to sample for each dataset

    Returns:
        dataloader for concat dataset
    """

    # Need a dataset with all features included for adgcl pre-training
    if for_adgcl:
        datasets, _ = get_chemical_datasets(transforms, -1, stage="train-adgcl")
        combined = []
        for data in datasets:
            combined += data
        return DataLoader(combined, batch_size=batch_size, shuffle=True)

    if "chemical" in subset:
        datasets, _ = get_chemical_datasets(transforms, num_social, stage="train")
    else:
        print("Skipping chemicals")
        datasets = []

    if "social" in subset:
        social_datasets, _ = get_social_datasets(transforms, num_social, stage="train", exclude=social_excludes)
    else:
        print("Skipping socials")
        social_datasets = []
    print(subset)

    if subset == ["dummy", "dummy"]:
        datasets, _ = get_random_datasets(transforms, num_social, stage = "train")
        social_datasets = []

    print(datasets, social_datasets)
    datasets += social_datasets
    combined = []
    # Concat dataset
    print(datasets)
    for data in datasets:
        combined += data

    return DataLoader(combined, batch_size=batch_size, shuffle=True)

def get_test_loaders(batch_size, transforms, num = 2000):
    """
    Get a list of validation loaders

    Args:
        dataset: the -starting dataset-, a hangover from previous code, likely to be gone in the next refactor
        batch_size: batch size for loaders
        transforms: a set of transforms applied to the data
        num: the maximum number of samples in each dataset (and therefore dataloader)

    Returns:
        datasets: list of dataloaders
        names: name of each loaders' respective dataset

    """

    datasets, names = get_test_datasets(transforms, num=num)
    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, names

def get_mol_test_loaders(batch_size, transforms, num = 2000):
    """
    Get a list of validation loaders

    Args:
        dataset: the -starting dataset-, a hangover from previous code, likely to be gone in the next refactor
        batch_size: batch size for loaders
        transforms: a set of transforms applied to the data
        num: the maximum number of samples in each dataset (and therefore dataloader)

    Returns:
        datasets: list of dataloaders
        names: name of each loaders' respective dataset

    """

    datasets, names = get_test_datasets(transforms, num=-1, mol_only = True)
    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, names

def get_mol_val_loaders(batch_size, transforms, num = 5000):
    """
    Get a list of validation loaders

    Args:
        batch_size: batch size for loaders
        transforms: a set of transforms applied to the data
        num: the maximum number of samples in each dataset (and therefore dataloader)

    Returns:
        datasets: list of dataloaders
        names: name of each loaders' respective dataset

    """

    datasets, names = get_val_datasets(transforms, num = -1, mol_only = True)
    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, names

def get_test_datasets(transforms, num = 2000, mol_only = False):

    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="test")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="test")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

def get_val_loaders(batch_size, transforms, num = 5000):
    """
    Get a list of validation loaders

    Args:
        batch_size: batch size for loaders
        transforms: a set of transforms applied to the data
        num: the maximum number of samples in each dataset (and therefore dataloader)

    Returns:
        datasets: list of dataloaders
        names: name of each loaders' respective dataset

    """

    datasets, names = get_val_datasets(transforms, num = num)
    datasets = [DataLoader(data, batch_size=batch_size) for data in datasets]

    return datasets, names

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

def get_train_datasets(transforms, num = 2000):
    
    chemical_datasets, ogbg_names = get_chemical_datasets(transforms, num, stage="train")
    if not mol_only:
        social_datasets, social_names = get_social_datasets(transforms, num, stage="train")
    else:
        social_datasets = []
        social_names = []

    datasets = chemical_datasets + social_datasets

    return datasets, ogbg_names + social_names

