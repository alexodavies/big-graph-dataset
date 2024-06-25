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
from utils import describe_one_dataset

if __name__ == "__main__":
    print(os.getcwd())

from utils import ESWR

def five_cycle_worker(g):
    """
    Returns the number of 5-cycles in a graph, normalised by the number of nodes
    """
    cycles = nx.simple_cycles(g, length_bound = 5)
    return len(list(cycles))

def vis_from_pyg(data, filename = None):
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    fig, ax = plt.subplots(figsize = (6,6))

    pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx_edges(g, pos = pos, ax = ax)
    nx.draw_networkx_nodes(g, pos = pos, node_color=labels, cmap="tab20",
                           vmin = 0, vmax = 20, ax = ax)

    ax.axis('off')

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def specific_from_networkx(graph):
    node_labels = []
    node_attrs = []
    edge_indices = []
    # Collect node labels and attributes
    for n in list(graph.nodes(data=True)):
        node_labels.append(n[1]["label"])
        node_attrs.append(n[1]["attrs"])

    # Collect edge indices and attributes
    for e in graph.edges(data=True):
        edge_indices.append((e[0], e[1]))

        # uncomment for edge attributes:
        # edge_attrs.append(e[2]["attr"]) 

    # Convert to PyTorch tensors
    node_labels = torch.stack(node_labels).flatten()

    # Specific to classification on nodes! Hard coding num classes as this happens on a per-graph basis
    node_labels = one_hot(node_labels.to(int), num_classes = 7)

    node_attrs = torch.stack(node_attrs)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    data = Data(x=node_attrs, edge_index=edge_indices, edge_attr = None,  y=node_labels)

    return data

def download_cora(visualise = False):
    # Ideally we'd save and reload graphs - but cora has massive feature dims
    zip_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"

    start_dir = os.getcwd()
    # print(os.getcwd(), os.listdir())
    os.chdir("bgd_files")

    if "cora" not in os.listdir():
        print("Downloading CORA")
        os.mkdir("cora")
        os.chdir("cora")
        _ = wget.download(zip_url)
    else:
        os.chdir("cora")
        if "cora_ml.npz" not in os.listdir():
            _ = wget.download(zip_url)

    edges = read_npz("cora_ml.npz")

    G = to_networkx(edges, to_undirected=True)

    node_classes = {n: int(edges.y[i].item()) for i, n in enumerate(list(G.nodes()))}
    node_attrs = {n: edges.x[i] for i, n in enumerate(list(G.nodes()))}
    # print(node_classes)

    # base_tensor = torch.Tensor([0])

    for node in list(G.nodes()):
        # class_tensor = base_tensor.clone()
        class_tensor = torch.Tensor([node_classes[node]])#node_classes[node]
        node_attr = node_attrs[node]

        # print(class_tensor, node_classes[node])

        G.nodes[node]["attrs"] = node_attr
        G.nodes[node]["label"] = class_tensor

    # for edge in list(G.edges()):
    #     G.edges[edge]["attrs"] = torch.Tensor([1])

    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    os.chdir(start_dir)

    return graph





def get_cora_dataset(num = 2000, targets = False):
    cora_graph = download_cora()
    nx_graph_list = ESWR(cora_graph, num, 96)

    data_objects = [specific_from_networkx(graph) for graph in tqdm(nx_graph_list, desc = "Converting back to pyg graphs")]

    return  data_objects

class CoraDataset(InMemoryDataset):
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
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])


    @property
    def raw_file_names(self):
        return ['cora_ml.npz']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Cora files exist")
            return

        data_list = get_cora_dataset(num=self.num, targets=self.stage != "train")

        new_data_list = []
        for i, item in enumerate(data_list):
            n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


            data = Data(x = item.x,
                        edge_index=item.edge_index,
                        edge_attr=None,
                        y = item.y)

            new_data_list.append(data)
        data_list = new_data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])


if __name__ == "__main__":
    dataset = CoraDataset(os.getcwd()+'/bgd_files/'+'cora', stage = "train")
    describe_one_dataset(dataset)


    dataset = CoraDataset(os.getcwd()+'/bgd_files/'+'cora', stage = "val")
    describe_one_dataset(dataset)
    dataset = CoraDataset(os.getcwd()+'/bgd_files/'+'cora', stage = "test")
    describe_one_dataset(dataset)