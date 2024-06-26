import os
from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from bgd.real.facebook_dataset import FacebookDataset
from bgd.real.ego_dataset import EgoDataset
from bgd.real.cora_dataset import CoraDataset
from bgd.real.neural_dataset import NeuralDataset
from bgd.real.road_dataset import RoadDataset
from bgd.real.reddit_dataset import RedditDataset
from bgd.real.from_ogb_dataset import FromOGBDataset, from_ogb_dataset
from bgd.real.from_tu_dataset import from_tu_dataset

from bgd.synthetic.random_dataset import RandomDataset
from bgd.synthetic.tree_dataset import TreeDataset
from bgd.synthetic.lattice_dataset import LatticeDataset
from bgd.synthetic.community_dataset import CommunityDataset


def get_datasets(transforms, num, stage="train", exclude=None, include=None):
    """
    Retrieves and transforms a list of datasets based on specified inclusion and exclusion criteria.

    Parameters:
    -----------
    transforms : function
        A function to apply transformations to each dataset.
    num : int
        The number of data points to include in each dataset.
    stage : str, optional
        The stage of data processing (e.g., "train", "test", "validate"). Default is "train".
    exclude : list or str, optional
        A list or a single string specifying dataset names to exclude from the selection.
        If None, no datasets will be excluded. Default is None.
    include : list or str, optional
        A list or a single string specifying dataset names to include in the selection.
        If None, all datasets not in the exclude list will be included. Default is None.

    Returns:
    --------
    datasets : list
        A list of transformed datasets.
    names : list
        A list of names of the selected datasets.

    Notes:
    ------
    - If both `exclude` and `include` are provided, the function first applies the `exclude` filter
      and then the `include` filter.
    - The function checks for the existence of a "bgd_files" directory and creates it if it does not exist.
    - The function supports various datasets, including predefined datasets and those from the Open Graph Benchmark (OGB) and TU datasets.

    Example:
    --------
    >>> def dummy_transform(dataset):
    >>>     return dataset
    >>> datasets, names = get_datasets(dummy_transform, num=100, stage="train", exclude=["reddit"], include=["cora", "trees"])
    >>> print(names)
    ['cora', 'trees']
    """

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

    names = list(all_datasets.keys())

    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        names = [d for d in names if d not in exclude]

    if include:
        if isinstance(include, str):
            include = [include]
        names = [d for d in names if d in include]

    datasets = [transforms(all_datasets[name](os.path.join(os.getcwd(), 'bgd_files', name), num=num, stage=stage))
                for name in names]

    return datasets, names

def get_node_task_datasets(transforms, num=5000, stage="train"):
    """
    Returns datasets with node-level tasks, both classification and regression.

    Args:
        transforms (list): List of data transformations to apply to the datasets.
        num (int, optional): Number of datasets to retrieve. Defaults to 5000.
        stage (str, optional): Stage of the datasets to retrieve. Defaults to "train".

    Returns:
        list: List of node task datasets (:obj:torch_geometric.data.InMemoryDataset).

    """
    includes = ["facebook_large", "cora"]
    return get_datasets(transforms, num, stage, include=includes)

def get_edge_task_datasets(transforms, num=5000, stage="train"):
    """
    Returns datasets with edge-level tasks, both regression and classification.

    Args:
        transforms (list): List of data transformations to apply to the datasets.
        num (int, optional): Number of datasets to retrieve. Defaults to 5000.
        stage (str, optional): Stage of the datasets to retrieve. Defaults to "train".

    Returns:
        list: List of edge task datasets (:obj:torch_geometric.data.InMemoryDataset).

    """
    includes = ["fruit_fly", "reddit"]
    return get_datasets(transforms, num, stage, include=includes)

def get_graph_task_datasets(transforms, num=5000, stage="train"):
    """
    Returns datasets with graph-level tasks, both regression and classification.

    Args:
        transforms (list): List of data transformations to apply to the datasets.
        num (int, optional): Number of datasets to retrieve. Defaults to 5000.
        stage (str, optional): Stage of the datasets to retrieve. Defaults to "train".

    Returns:
        list: List of graph task datasets (:obj:torch_geometric.data.InMemoryDataset).

    """
    non_graph_level_excludes = ["facebook_large", "cora", "fruit_fly", "reddit"]
    return get_datasets(transforms, num, stage, exclude=non_graph_level_excludes)

def get_graph_classification_datasets(transforms, num=5000, stage="train"):
    """
    Returns datasets with graph classification tasks.

    Args:
        transforms (list): List of data transformations to apply to the datasets.
        num (int, optional): Number of datasets to retrieve. Defaults to 5000.
        stage (str, optional): Stage of the datasets to retrieve. Defaults to "train".

    Returns:
        list: List of graph classification datasets (:obj:torch_geometric.data.InMemoryDataset).

    """
    non_graph_classification_excludes = ["facebook_large", "cora", "fruit_fly", "reddit",
                                         "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo",
                                         "community", "trees", "random"]
    return get_datasets(transforms, num, stage, exclude=non_graph_classification_excludes)

def get_graph_regression_datasets(transforms, num=5000, stage="train"):
    """
    Returns datasets with graph regression tasks.

    Args:
        transforms (list): List of data transformations to apply to the datasets.
        num (int, optional): Number of datasets to retrieve. Defaults to 5000.
        stage (str, optional): Stage of the datasets to retrieve. Defaults to "train".

    Returns:
        list: List of graph regression datasets (:obj:torch_geometric.data.InMemoryDataset).

    """
    non_graph_level_excludes = ["facebook_large", "cora", "fruit_fly", "reddit",
                                "ogbg-molpcba", "ogbg-molhiv", "ogbg-moltox21",
                                "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox",
                                "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast",
                                "MUTAG", "ENZYMES", "PROTEINS",
                                "COLLAB", "IMDB-BINARY", "REDDIT-BINARY"]
    return get_datasets(transforms, num, stage, exclude=non_graph_level_excludes)

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

