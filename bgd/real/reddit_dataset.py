import os
import pickle
import wget
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from utils import ESWR
import torch
from torch_geometric.data import Data, InMemoryDataset


def fix_property_string(input_string):
    input_string = input_string.split(',')
    input_string = [float(item) for item in input_string]

    return np.array(input_string)

def download_reddit():
    print("Getting reddit networkx graph")
    start_dir = os.getcwd()
    print(start_dir)
    os.chdir("bgd_files/reddit")
    print(os.getcwd())

    graph_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
    embedding_url = "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"
        
    if "reddit-graph.npz" in os.listdir():
        with open("reddit-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir(start_dir)
        return graph

    if "soc-redditHyperlinks-title.tsv" not in os.listdir():
        graph_data = wget.download(graph_url)
    if "web-redditEmbeddings-subreddits.csv" not in os.listdir():
        embeddings = wget.download(embedding_url)

    # We know that there are 300 components in the node feature vectors
    embedding_column_names = ["COMPONENT", *[i for i in range(300)]]
    embeddings = pd.read_csv("web-redditEmbeddings-subreddits.csv", names=embedding_column_names).transpose()
    graph_data = pd.read_csv("soc-redditHyperlinks-title.tsv", sep = "\t")


    # Avoids weird directory problems
    os.chdir(start_dir)

    embeddings.columns = embeddings.iloc[0]
    embeddings = embeddings.drop(["COMPONENT"], axis = 0)

    graph = nx.Graph()

    for col in tqdm(embeddings.columns, desc = "Adding nodes"):
        # attrs here is taken from the embedding data, with the node id the column (col)
        graph.add_node(col, attrs=embeddings[col].to_numpy().astype(float))

    sources = graph_data["SOURCE_SUBREDDIT"].to_numpy()
    targets = graph_data["TARGET_SUBREDDIT"].to_numpy()

    # This line can take a while!
    attrs = [fix_property_string(properties) for properties in tqdm(graph_data["PROPERTIES"].tolist(), desc = "Wrangling edge features")]
    labels = graph_data["LINK_SENTIMENT"].to_numpy()
    all_nodes = list(graph.nodes())

    for i in tqdm(range(sources.shape[0]), desc = "Adding edges"):
        if sources[i] in all_nodes and targets[i] in all_nodes:
            graph.add_edge(sources[i], targets[i],
                        labels = labels[i],
                        attr = attrs[i])


    # Last tidying bits
    graph = nx.convert_node_labels_to_integers(graph)
    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    # Save the graph!
    with open("bgd_files/reddit/reddit-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    return graph

def get_reddit_dataset(num = 2000):
    graph = download_reddit()

    # Sample 1000 graphs of max 96 nodes from the big reddit graph
    nx_graph_list = ESWR(graph, 1000, 96)

    pyg_graph_list = [specific_from_networkx(g) for g in nx_graph_list]

    return pyg_graph_list

def specific_from_networkx(graph):
    # Turns a graph into a pytorch geometric object
    # Mostly by unpacking dictionaries on nodes and edges
    # Here node labels are the target
    # One of these functions for each dataset ideally - they are unlikely to transfer across datasets
    node_attrs = []
    edge_indices = []
    edge_labels = []
    edge_attrs = []

    # Collect node labels and attributes
    for n in list(graph.nodes(data=True)):
        # list(graph.nodes(data=True)) returns [(node_id1, {attribute dictionary}), (node_id2, ...), (node_id3, ...)]
        node_attrs.append(torch.Tensor(n[1]["attrs"]))

    # Collect edge indices and attributes
    for e in graph.edges(data=True):
        # graph.edges(data=True) is a generator producing (node_id1, node_id2, {attribute dictionary})
        edge_indices.append((e[0], e[1]))

        edge_attrs.append(torch.Tensor(e[2]["attr"])) 
        edge_labels.append(e[2]["labels"])


    # Specific to classification on edges! This is a binary edge classification (pos/neg) task
    edge_labels = ((torch.Tensor(edge_labels) + 1)/2).reshape(-1,1)

    edge_attrs = torch.stack(edge_attrs)
    node_attrs = torch.stack(node_attrs)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    # Can pass:
    # x:            node features, shape (n nodes x n features)
    # edge_index:   the list of edges in the graph, shape (2, n_edges). Entries edge_index[i, :] are [node_id1, node_id2].
    # edge_attr:    edge features, shape (n_edges, n_features), same order as edgelist
    # y:            targets. Graph regression shape (n_variables), graph classification (n_classes), node classification (n_nodes, n_classes), edge classification (n_edges, n_classes)
    data = Data(x=node_attrs, edge_index=edge_indices, edge_attr = edge_attrs,  y=edge_labels)

    return data


class RedditDataset(InMemoryDataset):
    r"""
    Contributor: Alex O. Davies
    
    Contributor email: `alexander.davies@bristol.ac.uk`
    

    Reddit hyperlink graphs - ie graphs of subreddits interacting with one another.
    The original graph is sourced from:

         `Kumar, Srijan, et al. "Community interaction and conflict on the web." Proceedings of the 2018 world wide web conference. 2018.`

    We produce this dataset of small graphs using ESWR.
    The data has text embeddings as node features for each subreddit and text features for the cross-post edges.

    The task is edge classification for the sentiment of the interaction between subreddits.

     - Task: Edge classification
     - Num node features: 300
     - Num edge features: 86
     - Num target values: 1
     - Target shape: N Edges
     - Num graphs: Parameterised by `num`

    Args:
        root (str): Root directory where the dataset should be saved.
        stage (str): The stage of the dataset to load. One of "train", "val", "test". (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        num (int): The number of samples to take from the original dataset. (default: :obj:`2000`).
    """

    def __init__(self, root,  stage = "train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        
        # Options are node-classification, node-regression, graph-classification, graph-regression, edge-regression, edge-classification
        # Graph-level tasks are preferred! (graph-classification and graph-regression)
        # edge-prediction is another option if you can't think of a good task
        self.task = "edge-classification"

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])


    @property
    def raw_file_names(self):
        # Replace with your saved raw file name
        return ['reddit-graph.npz']

    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print(f"Reddit files exist")
            return

        # Get a list of num pytorch_geometric.data.Data objects
        data_list = get_reddit_dataset(self.num)

        # Torch geometric stuff
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.stage_to_index[self.stage]])