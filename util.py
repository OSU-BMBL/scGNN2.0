import os

import numpy as np
import scipy.sparse as sp
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset

def generateCelltypeRegu(listResult):
    celltypesample = np.zeros((len(listResult), len(listResult)))
    tdict = {}
    count = 0
    for item in listResult:
        if item in tdict:
            tlist = tdict[item]
        else:
            tlist = []
        tlist.append(count)
        tdict[item] = tlist
        count += 1

    for key in sorted(tdict):
        tlist = tdict[key]
        for item1 in tlist:
            for item2 in tlist:
                celltypesample[item1, item2] = 1.0

    return celltypesample


class ExpressionDataset(Dataset):
    def __init__(
        self, 
        X=None, 
        transform=None
        ):
        """
        Args:
            X : ndarray (dense) or list of lists (sparse) [cell * gene]
            transform (callable, optional): apply transform function if not none
        """
        self.X = X # [cell * gene]

        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0] # of cell

    def __getitem__(self, idx):
        
        # Get sample (one cell)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        # Convert to Tensor
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample, idx

class ClusterDataset(ExpressionDataset):
    def __init__(self, X=None, transform=None):
        super().__init__(X, transform)


def plot(y, xlabel='epochs', ylabel='', hline=None, output_dir='', suffix=''):
    plt.plot(range(len(y)), y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=hline, color='green', linestyle='-') if hline else None
    plt.savefig(os.path.join(output_dir, f"{ylabel.replace(' ', '_')}{suffix}.png"), dpi=200)
    plt.clf()

def drawUMAP(z, listResult, output_dir):
    """
    UMAP
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(z)
    size = len(set(listResult)) + 1

    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=listResult, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(int(size)) - 0.5).set_ticks(np.arange(int(size)))
    plt.title('UMAP projection', fontsize=24)

    plt.savefig(os.path.join(output_dir, f"UMAP.png"), dpi=300)
    plt.clf()

def drawTSNE(z, listResult, output_dir):
    size = len(set(listResult))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(z)

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['Cluster'] = listResult
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="Cluster",
        palette=sns.color_palette("brg", int(size)),
        data=df_subset,
        legend="full",
        # alpha=0.3
    )
    
    plt.savefig(os.path.join(output_dir, f"tSNE.png"), dpi=300)
    plt.clf()

def imputation_err_heatmap(X_sc, X_imputed, cluster_labels=None, args=None):
    pass