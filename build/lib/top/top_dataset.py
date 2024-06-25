import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from torch.utils.data.dataset import ConcatDataset


class ToPDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    Contributor email: `alexander.davies@bristol.ac.uk`
    
    Processes an InMemoryDataset into a ToP dataset by removing node and edge features.
    
    Based on the paper:

         `Towards Generalised Pre-Training of Graph Models, Davies, A. O., Green, R. W., Ajmeri, N. S., and Silva Filho, T. M.,  arXiv e-prints, 2024. doi:10.48550/arXiv.2311.03976.`

    The resulting dataset is topology-only, intended for pre-training with ToP, and as such this module does not produce validation/test splits.

    Saves a single processed file `train-top.pt` under `./root/processed/`.

    Args:
        original_dataset (InMemoryDataset): The original dataset to convert to ToP format.
        root (str): Root directory where the dataset should be saved. If 'none', will use the root directory of `original_dataset`. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. `num=-1` will convert all available samples from the original. (default: :obj:`-1`).
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)

    """

    def __init__(self, original_dataset, root = None,  num = -1, transform=None, pre_transform=None, pre_filter=None):
        self.original_dataset = original_dataset
        self.stage = "train"
        self.stage_to_index = {"train":0}
        self.num = num
        root = root if root is not None else original_dataset.root
        print(f"\n\nConverting original dataset {original_dataset} at {root}")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['dummy.csv']

    @property
    def processed_file_names(self):
        return ['train-top.pt']


    def process(self):
        # Read data into huge `Data` list.
        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print(f"Processed ToP files exist at {self.processed_paths[self.stage_to_index[self.stage]]}")
            return
        
        data_list = self.original_dataset# get_fb_dataset(num=self.num)
        new_data_list = []
        if type(data_list) == ConcatDataset:
            print("Found concat dataset in ToP conversion")
            for dataset in data_list.datasets:
                for item in dataset:
                    new_data_list.append(item)
            self.original_dataset = new_data_list
            data_list = new_data_list

        num_samples = len(data_list)
        if self.num == -1 or num_samples < self.num:
            keep_n = num_samples
        else:
            keep_n = self.num

        print(f"Dropping targets from {keep_n} samples out of {num_samples}")
        new_data_list = []
        for i, item in enumerate(tqdm(data_list[:keep_n], desc = "Converting dataset to ToP")):

            if item.x is not None:
                n_nodes = item.x.shape[0]
            else:
                n_nodes = torch.max(item.edge_index) + 1

            n_edges =  item.edge_index.shape[1]
            data = Data(x = torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
                        edge_index=item.edge_index,
                        edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                        y = None)

            new_data_list.append(data)
        data_list = new_data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])
