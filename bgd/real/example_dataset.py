import os
import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.io import read_npz
from torch.nn.functional import one_hot
import wget
import matplotlib.pyplot as plt
from utils import describe_one_dataset, vis_grid, ESWR
from utils import ESWR



def specific_from_networkx(graph):
    # Turns a graph into a pytorch geometric object
    # Mostly by unpacking dictionaries on nodes and edges
    # Here node labels are the target
    # One of these functions for each dataset ideally - they are unlikely to transfer across datasets
    node_labels = []
    node_attrs = []
    edge_indices = []

    # Collect node labels and attributes
    for n in list(graph.nodes(data=True)):
        # list(graph.nodes(data=True)) returns [(node_id1, {attribute dictionary}), (node_id2, ...), (node_id3, ...)]
        node_labels.append(n[1]["label"])
        node_attrs.append(n[1]["attrs"])

    # Collect edge indices and attributes
    for e in graph.edges(data=True):
        # graph.edges(data=True) is a generator producing (node_id1, node_id2, {attribute dictionary})
        edge_indices.append((e[0], e[1]))

        # uncomment for edge attributes:
        # edge_attrs.append(e[2]["attrs"]) 

    # Convert to PyTorch tensors
    node_labels = torch.stack(node_labels).flatten()

    # Specific to classification on nodes! Hard coding num classes as this happens on a per-graph basis
    # Per-graph classification should also be one-hot
    node_labels = one_hot(node_labels.to(int), num_classes = 7)

    node_attrs = torch.stack(node_attrs)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    # Can pass:
    # x:            node features, shape (n nodes x n features)
    # edge_index:   the list of edges in the graph, shape (2, n_edges). Entries edge_index[i, :] are [node_id1, node_id2].
    # edge_attr:    edge features, shape (n_edges, n_features), same order as edgelist
    # y:            targets. Graph regression shape (n_variables), graph classification (n_classes), node classification (n_nodes, n_classes), edge classification (n_edges, n_classes)
    data = Data(x=node_attrs, edge_index=edge_indices, edge_attr = None,  y=node_labels)

    return data

def download_example():
    # This function should download the data and process it into one big networkx graph

    # If node labels are the target, they should be included as an attribute of each node
    # In this case each node is (id, {"attrs":..., "label":...})
    # Probably the same is best for edge labels

    # url to data
    zip_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"

    # swap into directory
    start_dir = os.getcwd()
    os.chdir("bgd_files")

    # create directory if we haven't before
    if "example" not in os.listdir():
        print("Downloading CORA (as an example!)")
        os.mkdir("example")
        os.chdir("example")
    # Means that we've previously downloaded data
    else:
        os.chdir("example")

    if "cora_ml.npz" not in os.listdir():
        _ = wget.download(zip_url)
    edges = read_npz("cora_ml.npz")

    G = to_networkx(edges, to_undirected=True)

    node_classes = {n: int(edges.y[i].item()) for i, n in enumerate(list(G.nodes()))}
    node_attrs = {n: edges.x[i] for i, n in enumerate(list(G.nodes()))}

    for node in list(G.nodes()):
        class_tensor = torch.Tensor([node_classes[node]])
        node_attr = node_attrs[node]

        # If you have node features:
        G.nodes[node]["attrs"] = node_attr

        # If you have node labels:
        G.nodes[node]["label"] = class_tensor

        # If you don't have any node features:
        # G.nodes[node]["attrs"] = torch.Tensor([1])

    # No edge features for CORA, uncomment if you need
    # for edge in list(G.edges()):
    #     G.edges[edge]["attrs"] = torch.Tensor([1])
    #     G.edges[edge]["label"] = torch.Tensor([1])

    # This is generic cleaning stuff - we take the largest connected component and re-set node ids
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    # Important! Need to move back to start directory for other code
    os.chdir(start_dir)

    return graph

def get_example_dataset(num = 2000):
    # Get the cora graph
    cora_graph = download_example()

    # ESWR samples a load of graphs from a large input graph
    # Doesn't handle graph labels! See download_example for how to deal with this
    # Arguments are networkx.Graph, num graphs to sample, size doesn't currently do anything
    nx_graph_list = ESWR(cora_graph, num, 96)

    data_objects = [specific_from_networkx(graph) for graph in tqdm(nx_graph_list, desc = "Converting back to pyg graphs")]

    return  data_objects

class ExampleDataset(InMemoryDataset):
    # Documentation is essential! Without the sources listed I won't be able to include the dataset
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    

    Academic citation graphs from the ML community, sampled from a large original graph using ESWR.
    The original graph is sourced from:

         `Yang, Zhilin, William Cohen, and Ruslan Salakhudinov. "Revisiting semi-supervised learning with graph embeddings." International conference on machine learning. PMLR, 2016.`

    The original data has one-hot bag-of-words over paper abstract as node features.

    The task is node classification for the category of each paper, one-hot encoded for seven categories.

     - Task: Node classification
     - Num node features: 2879
     - Num edge features: None
     - Num target values: 7
     - Target shape: N Nodes
     - Num graphs: Parameterised by `num`

    Args:
        root (str): Root directory where the dataset should be saved.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. (default: :obj:`2000`).
    """
    def __init__(self, root, stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}

        # Options are node-classification, node-regression, graph-classification, graph-regression, edge-regression, edge-classification
        # Graph-level tasks are preferred! (graph-classification and graph-regression)
        # edge-prediction is another option if you can't think of a good task
        self.task = "node-classification"

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])


    @property
    def raw_file_names(self):
        # Replace with your saved raw file name
        return ['cora_ml.npz']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print(f"Cora files exist")
            return

        # Get a list of num pytorch_geometric.data.Data objects
        data_list = get_example_dataset(num=self.num)

        # You can iterate over the data objects if necessary:
        # ===================================================
        # new_data_list = []
        # for i, item in enumerate(data_list):
        #     n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

        #     data = Data(x = item.x,
        #                 edge_index=item.edge_index,
        #                 # Here we don't have any edge features so we use just 1s
        #                 edge_attr=None,
        #                 y = item.y)

        #     new_data_list.append(data)
        # data_list = new_data_list
        # ===================================================

        # Torch geometric stuff
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


# Run with python -m datasets/example_dataset.py
# Run this to check that everything works!
if __name__ == "__main__":
    # Please set the last part of the path to your dataset name!
    dataset = ExampleDataset(os.getcwd()+'/bgd_files/'+'example', stage = "train")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/example/train.png")

    dataset = ExampleDataset(os.getcwd()+'/bgd_files/'+'example', stage = "val")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/example/val.png")

    dataset = ExampleDataset(os.getcwd()+'/bgd_files/'+'example', stage = "test")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/bgd_files/example/test.png")