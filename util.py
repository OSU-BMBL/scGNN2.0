import os

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
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
    plt.savefig(os.path.join(output_dir, f"{ylabel.replace(' ', '_')}{suffix}.png"))
    plt.clf()
