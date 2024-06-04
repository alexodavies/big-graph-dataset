import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from torch.utils.data.dataset import ConcatDataset


class ToPDataset(InMemoryDataset):
    def __init__(self, root, original_dataset, stage = "train", num = -1, transform=None, pre_transform=None, pre_filter=None):
        self.original_dataset = original_dataset
        self.stage = stage
        self.stage_to_index = {"train":0}
        self.num = num
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
