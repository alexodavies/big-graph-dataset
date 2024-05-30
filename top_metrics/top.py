import logging
import random

import numpy as np
import torch
import yaml

from unsupervised.encoder import Encoder
from unsupervised.learning import GInfoMinMax
from torch_geometric.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler

from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False, every = 1, node_features = False):
    with torch.no_grad():
        x, y = encoder.get_embeddings(loader, device, is_rand_label, every = every, node_features = node_features)
    if dtype == 'numpy':
        return x,y
    elif dtype == 'torch':
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    else:
        raise NotImplementedError

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if data is None:
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        # For data between -1 and 1, the middle is 0
        threshold = im.norm(0)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

class GeneralEmbeddingEvaluation():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def embedding_evaluation(self, encoder, train_loaders, names):
        train_all_embeddings, train_separate_embeddings = self.get_embeddings(encoder, train_loaders)
        self.centroid_similarities(train_separate_embeddings, names)
        self.vis(train_all_embeddings, train_separate_embeddings, names)

    def get_embeddings(self, encoder, loaders):
        encoder.eval()
        all_embeddings = None
        separate_embeddings = []
        # colours = []
        for i, loader in enumerate(tqdm(loaders, leave = False, desc = "Getting embeddings")):
            train_emb, train_y = get_emb_y(loader, encoder, self.device, is_rand_label=False, every=1, node_features = False)
            
            separate_embeddings.append(train_emb)
            if all_embeddings is None:
                all_embeddings = train_emb
            else:
                all_embeddings = np.concatenate((all_embeddings, train_emb))
            # colours += [i for n in range(train_emb.shape[0])]

        scaler = StandardScaler().fit(all_embeddings)
        for embedding in separate_embeddings:
            embedding = scaler.transform(embedding)
            embedding = normalize(embedding)

        all_embeddings = scaler.transform(all_embeddings)
        all_embeddings = normalize(all_embeddings)
        
        return all_embeddings, separate_embeddings

    def vis(self, all_embeddings, separate_embeddings, names):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 9))

        cmap = plt.get_cmap('viridis')
        unique_categories = np.unique(names)
        colors = cmap(np.linspace(0, 1, len(unique_categories)))
        color_dict = {category: color for category, color in zip(unique_categories, colors)}

        cmap = plt.get_cmap('autumn')
        unique_categories = np.unique(names)
        colors = cmap(np.linspace(0, 1, len(unique_categories)))
        mol_color_dict = {category: color for category, color in zip(unique_categories, colors)}

        embedder = UMAP(n_components=2, n_jobs=4, n_neighbors = 30).fit(all_embeddings)

        for i, emb in enumerate(separate_embeddings):
            proj = embedder.transform(emb)
            name = names[i]
            if "ogb" in name:
                plot_marker = "^"
                color = mol_color_dict[name]
            else:
                plot_marker = "x"
                color = color_dict[name]

            ax1.scatter(proj[:, 0], proj[:, 1],
                        alpha= 1 - proj.shape[0] / all_embeddings.shape[0], s = 5,
                        label=f"{names[i]}", # - {proj.shape[0]} graphs",
                        c = color, marker = plot_marker)

        ax1.legend(shadow=True)
        ax1.set_title("UMAP Embedding")

        embedder = PCA(n_components=2).fit(all_embeddings)


        for i, emb in enumerate(separate_embeddings):
            proj = embedder.transform(emb)
            name = names[i]
            if "ogb" in name:
                plot_marker = "^"
                color = mol_color_dict[name]
            else:
                plot_marker = "x"
                color = color_dict[name]

            ax2.scatter(proj[:, 0], proj[:, 1],
                        alpha= 1 - proj.shape[0] / all_embeddings.shape[0], s = 5,
                        c = color, marker = plot_marker)
            
                # Get the legend handles and labels

        ax2.set_title("PCA Projection")

        handles, labels = ax2.get_legend_handles_labels()

        # Create a new legend with increased size for scatter points
        new_handles = []
        for ihandle, handle in enumerate(handles):
            new_handle = plt.Line2D([0], [0],
             marker="^" if "ogb" in labels[ihandle] else "o", color='w',
               markerfacecolor=handle.get_facecolor()[0], markersize=30)
            new_handles.append(new_handle)

        # Add legend to the axis
        # ax.legend(handles=new_handles, labels=labels)

        fig.legend(handles = new_handles, labels = labels,
                   bbox_to_anchor=(0.05, -0.2), loc='lower left',
                     ncol = 8, frameon = False)
        
        
        
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig("outputs/embedding.png")
        plt.close()


    def centroid_similarities(self, embeddings, names):
        embed_dim = embeddings[0].shape[1]
        centroids = np.zeros((len(embeddings), embed_dim))

        for i, embedding in enumerate(embeddings):
            centroids[i, :] = np.mean(embedding, axis = 0)

        pairwise_similarities = cosine_similarity(centroids)
        print(pairwise_similarities)

        fig, ax = plt.subplots(figsize=(7,6))

        im = ax.imshow(pairwise_similarities, cmap = "binary", vmin = -1, vmax = 1)

        ax.set_xticks(np.arange(len(names)), labels = names)
        ax.set_yticks(np.arange(len(names)), labels = names)

        annotate_heatmap(im, valfmt="{x:.3f}")

        plt.savefig("outputs/pairwise-similarity.png")


        # pairwise_sum = 0
        # for i1 in range(pairwise_similarities.shape[0]):
        #     for i2 in range(pairwise_similarities.shape[1]):
        #         if i2 <= i1:
        #             pass
        #         else:
        #             pairwise_sum += pairwise_similarities[i1, i2]

        # mean_separation = pairwise_sum / ((pairwise_similarities.shape[0]**2)/2 - pairwise_similarities.shape[0])
    plt.show()

def compute_scores(datasets, names):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup_seed(args.seed)

    checkpoint = "all-100.pt"

    checkpoint_path = f"outputs/{checkpoint}"
    cfg_name = checkpoint.split('.')[0] + ".yaml"
    config_path = f"outputs/{cfg_name}"

    with open(config_path, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandb_cfg = yaml.safe_load(stream)

            # Printing dictionary
            print(wandb_cfg)
        except yaml.YAMLError as e:
            print(e)

    args = wandb_cfg

    # Get datasets
    train_loaders = [DataLoader(data, batch_size=128) for data in datasets]
    
    model = GInfoMinMax(
        Encoder(emb_dim=args["emb_dim"]["value"], num_gc_layers=args["num_gc_layers"]["value"], drop_ratio=args["drop_ratio"]["value"],
                pooling_type="standard"),
        proj_hidden_dim=args["emb_dim"]["value"]).to(device)

    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['encoder_state_dict'])


    # Get embeddings
    general_ee = GeneralEmbeddingEvaluation()
    model.eval()

    general_ee.embedding_evaluation(model.encoder, train_loaders, names)

    # for i_embedding, embedding in enumerate(val_separate_embeddings):
    #     val_loader = val_loaders[i_embedding]
    #     test_loader = test_loaders[i_embedding]
    #     test_embedding = test_separate_embeddings[i_embedding]
    #     name = names[i_embedding]
            
