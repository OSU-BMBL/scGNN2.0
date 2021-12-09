
"""
    1) Feature AE
    2) GAE
    3) Cluster AE
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

from auto_encoders.gat import GAT
from auto_encoders.gae.layers import GraphConvolution

class Feature_AE(nn.Module):
    ''' 
    Autoencoder for dimensional reduction
    Args:
        x: Tensor, mini-batch
        dim: int, feature dimension
    Return:
        self.decode(z): reconstructed input 
        z: feature encoding
    '''
    def __init__(self, dim):
        super(Feature_AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4(h3))

        # h3 = torch.sigmoid(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return z, self.decode(z)

class Graph_AE(nn.Module):
    def __init__(self, dim, embedding_size):
        super(Graph_AE, self).__init__()
        self.gat = GAT(num_of_layers = 2, 
                        num_heads_per_layer = [2, 2], 
                        num_features_per_layer = [dim, 64, 32])

        self.gc1 = GraphConvolution(dim, 32, 0, act=F.relu)
        self.gc2 = GraphConvolution(32, embedding_size, 0, act=lambda x: x)
        self.gc3 = GraphConvolution(32, embedding_size, 0, act=lambda x: x)

        self.decode = InnerProductDecoder(0, act=lambda x: x)

    def encode_gat(self, in_nodes_features, edge_index):
        return self.gat((in_nodes_features, edge_index))

    def encode_gae(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, in_nodes_features, edge_index, encode=False, use_GAT=True):
        # [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        gae_info= None
        
        if use_GAT:
            out_nodes_features = self.encode_gat(in_nodes_features, edge_index)[0]
        else:
            gae_info = self.encode_gae(in_nodes_features, edge_index)
            out_nodes_features = self.reparameterize(*gae_info)
        
        recon_graph = self.decode(out_nodes_features)
        return out_nodes_features, gae_info, recon_graph

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Cluster_AE(Feature_AE):
    def __init__(self, dim):
        super().__init__(dim)