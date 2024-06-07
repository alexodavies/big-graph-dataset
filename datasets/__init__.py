# from .loaders import get_test_datasets, get_val_datasets, get_train_datasets, get_all_datasets
# from .cora_dataset import CoraDataset
# from .road_dataset import RoadDataset
# from .neural_dataset import NeuralDataset
# from .facebook_dataset import FacebookDataset
# from .reddit_dataset import RedditDataset
# from .ego_dataset import EgoDataset
# from .random_dataset import RandomDataset
# from .community_dataset import CommunityDataset
# from .tree_dataset import TreeDataset

from datasets import real
from datasets import synthetic
from datasets import loaders

from .loaders import *
from .real import *
from .synthetic import *

__all__ = ['CoraDataset', 'RoadDataset', 'NeuralDataset',
            'FacebookDataset', 'RedditDataset', 'EgoDataset',
            'RandomDataset', 'CommunityDataset', 'TreeDataset',
              'get_test_datasets', 'get_val_datasets', 'get_train_datasets', 'get_all_datasets',
              'real', 'synthetic', 'loaders']