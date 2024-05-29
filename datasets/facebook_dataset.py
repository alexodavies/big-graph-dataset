import os
import networkx as nx
import pandas as pd
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import InMemoryDataset, Data
import pickle
import zipfile
import wget
import numpy as np
from utils import ESWR
import json

from littleballoffur.exploration_sampling import *

from utils import describe_one_dataset, vis_grid

print(os.getcwd())
# from utils import vis_from_pyg


# def download_reddit(visualise = False):
#     graph_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
#     embedding_url = "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

#     start_dir = os.getcwd()
#     # for _ in range(3):
#     #     os.chdir('../')
#     # print(os.getcwd(), os.listdir())
#     os.chdir("original_datasets")

#     if "reddit-graph.npz" in os.listdir():
#         with open("reddit-graph.npz", "rb") as f:
#             graph = pickle.load(f)
#         os.chdir('../')
#         return graph

#     if "soc-redditHyperlinks-title.tsv" not in os.listdir():
#         graph_data = wget.download(graph_url)
#     if "web-redditEmbeddings-subreddits.csv" not in os.listdir():
#         embedding_data = wget.download(embedding_url)


#     embedding_column_names = ["COMPONENT", *[i for i in range(300)]]
#     embeddings = pd.read_csv("web-redditEmbeddings-subreddits.csv", names=embedding_column_names).transpose()
#     graph_data = pd.read_csv("soc-redditHyperlinks-title.tsv", sep = "\t")

#     embeddings.columns = embeddings.iloc[0]
#     embeddings = embeddings.drop(["COMPONENT"], axis = 0)


#     graph = nx.Graph()

#     for col in embeddings.columns:
#         graph.add_node(col, attrs=embeddings[col].to_numpy().astype(float))

#     sources = graph_data["SOURCE_SUBREDDIT"].to_numpy()
#     targets = graph_data["TARGET_SUBREDDIT"].to_numpy()

#     for i in range(sources.shape[0]):
#         graph.add_edge(sources[i], targets[i])

#     for node in list(graph.nodes(data=True)):
#         data = node[1]
#         if len(data) == 0:
#             graph.remove_node(node[0])

#     graph = nx.convert_node_labels_to_integers(graph)
#     CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
#     CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
#     graph = CGs[0]
#     graph = nx.convert_node_labels_to_integers(graph)

#     with open("reddit-graph.npz", "wb") as f:
#         pickle.dump(graph, f)

#     os.chdir(start_dir)

#     return graph

def download_facebook(visualise = False):
    zip_url = "https://snap.stanford.edu/data/facebook_large.zip"

    start_dir = os.getcwd()
    # print(os.getcwd(), os.listdir())
    os.chdir("original_datasets")

    if "facebook-graph.npz" in os.listdir():
        with open("facebook-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph
    # print(os.getcwd())

    if "facebook_large" not in os.listdir():
        print("Downloading FB graph")
        _ = wget.download(zip_url)
        with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("facebook_large.zip")
    os.chdir("facebook_large")

    edgelist = pd.read_csv("musae_facebook_edges.csv")

    labels = pd.read_csv("musae_facebook_target.csv")

    with open("musae_facebook_features.json", 'r', encoding='utf-8') as file:
            features = json.load(file)



    # print(labels.head())
    # print(np.unique(labels["page_type"]))

    conversion_dict = {"company":       torch.Tensor([0]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "government":    torch.Tensor([1]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "politician":    torch.Tensor([2]), #, 0, 0, 0, 0, 0, 0, 0, 0]),
                       "tvshow":        torch.Tensor([3])} #, 0, 0, 0, 0, 0, 0, 0, 0]),


    graph = nx.Graph()
    label_specific = labels["page_type"]
    for col in labels["id"]:
        graph.add_node(int(col))
        # Turns out the facebook data has attributes of varying length
        graph.nodes[int(col)]["attrs"] = torch.Tensor([1]) # features[str(col)])
        graph.nodes[int(col)]["label"] = conversion_dict[label_specific[col]]
    sources = edgelist["id_1"].to_numpy().astype("int")
    targets = edgelist["id_2"].to_numpy().astype("int")

    for i in range(sources.shape[0]):
        graph.add_edge(sources[i], targets[i], attr = torch.Tensor([1]))

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)
    # print(f"Facebook {graph}")

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # with open("reddit-graph.npz", "wb") as f:
    #     pickle.dump(graph, f)

    os.chdir(start_dir)
    # print(graph)
    # quit()
    return graph

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
    node_labels = one_hot(node_labels.to(int), num_classes = 4)

    node_attrs = torch.stack(node_attrs)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    data = Data(x=node_attrs, edge_index=edge_indices, edge_attr = None,  y=node_labels)

    return data

def get_fb_dataset(num = 2000, targets = False):
    fb_graph = download_facebook()
    # print(fb_graph.nodes(data=True))
    nx_graph_list = ESWR(fb_graph, num, 96)


    data_objects = [specific_from_networkx(g) for g in nx_graph_list]

    return data_objects# loader

class FacebookDataset(InMemoryDataset):
    def __init__(self, root, stage="train", transform=None, pre_transform=None, pre_filter=None, num = 2000):
        self.num = num
        self.stage = stage
        self.stage_to_index = {"train":0,
                               "val":1,
                               "test":2}
        
        self.task = "node-classification"

        # _ = download_facebook()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.stage_to_index[self.stage]])

    @property
    def raw_file_names(self):
        return ['musae_facebook_edges.csv',
                'musae_facebook_target.csv']


    @property
    def processed_file_names(self):
        return ['train.pt',
                'val.pt',
                'test.pt']


    def process(self):
        # Read data into huge `Data` list.

        if os.path.isfile(self.processed_paths[self.stage_to_index[self.stage]]):
            print("Facebook files exist")
            return

        data_list = get_fb_dataset(num=self.num, targets=self.stage != "train")

        if self.stage == "train":
            print("Found stage train, dropping targets")
            new_data_list = []
            for i, item in enumerate(data_list):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]

                data = Data(x = torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
                            edge_index=item.edge_index,
                            edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
                            y = None)

                new_data_list.append(data)
            data_list = new_data_list
        else:
            new_data_list = []
            for i, item in enumerate(data_list):
                n_nodes, n_edges = item.x.shape[0], item.edge_index.shape[1]


                data = Data(x = item.x,# torch.ones(n_nodes).to(torch.int).reshape((-1, 1)),
                            edge_index=item.edge_index,
                            edge_attr=torch.ones(n_edges).to(torch.int).reshape((-1,1)),
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
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', stage = "train")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/original_datasets/facebook_large/train.png")
    
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', stage = "val")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/original_datasets/facebook_large/val.png")
    
    dataset = FacebookDataset(os.getcwd()+'/original_datasets/'+'facebook_large', stage = "test")
    describe_one_dataset(dataset)
    vis_grid(dataset[:16], os.getcwd()+"/original_datasets/facebook_large/test.png")