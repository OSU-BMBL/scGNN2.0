"""
"""

import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from scipy.spatial import distance
import scipy.sparse as sp
import networkx as nx

import auto_encoders.gae.utils as gae_util
import auto_encoders.gae.optimizer as gae_optimizer

import auto_encoders.model as model
import info_log

def graph_AE_handler(X_embed, CCC_graph, args, param):
    info_log.print('--------> Starting Graph AE ...')

    use_GAT = args.graph_AE_use_GAT
    learning_rate = args.graph_AE_learning_rate
    total_epoch = args.graph_AE_epoch

    # Prepare matrices
    if use_GAT:
        X_embed_normalized = normalize_features_dense(X_embed)
        X_embed_normalized = torch.from_numpy(X_embed_normalized).type(torch.FloatTensor).to(param['device'])

        CCC_graph_edge_index = convert_adj_to_edge_index(CCC_graph)
        CCC_graph_edge_index = torch.from_numpy(CCC_graph_edge_index).type(torch.LongTensor).to(param['device'])

        CCC_graph = torch.from_numpy(CCC_graph).type(torch.FloatTensor).to(param['device'])
    else:
        adj, adj_train, edgeList = feature2adj(X_embed)
        adj_norm = gae_util.preprocess_graph(adj)
        adj_label = (adj_train + sp.eye(adj_train.shape[0])).toarray()

        zDiscret = X_embed > np.mean(X_embed, axis=0)
        zDiscret = 1.0 * zDiscret
        X_embed_normalized = torch.from_numpy(zDiscret).type(torch.FloatTensor).to(param['device'])
        CCC_graph_edge_index = adj_norm.to(param['device'])
        CCC_graph = torch.from_numpy(adj_label).type(torch.FloatTensor).to(param['device'])

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    graph_AE = model.Graph_AE(X_embed.shape[1]).to(param['device'])
    optimizer = optim.Adam(graph_AE.parameters(), lr=learning_rate)

    for epoch in range(total_epoch):
        graph_AE.train()
        optimizer.zero_grad()

        embed, gae_info, recon_graph = graph_AE(X_embed_normalized, CCC_graph_edge_index, use_GAT=use_GAT)

        if use_GAT:
            loss = loss_function(preds = recon_graph,
                                labels = CCC_graph)
        else:
            loss = gae_optimizer.loss_function(preds = recon_graph,
                                            labels = CCC_graph,
                                            mu=gae_info[0], logvar=gae_info[1],
                                            n_nodes = X_embed.shape[0],
                                            norm = norm, 
                                            pos_weight = pos_weight)
        
        # Backprop and Update
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        info_log.interval_print(f"----------------> Epoch: {epoch+1}/{total_epoch}, Current loss: {cur_loss:.4f}", epoch=epoch, total_epoch=total_epoch)

    return embed.detach().cpu().numpy(), recon_graph.detach().cpu().numpy(), edgeList, adj # edgeList added just for benchmark testing

def loss_function(preds, labels):
    return F.binary_cross_entropy_with_logits(preds, labels)

def normalize_features_dense(node_features_dense):
    assert isinstance(node_features_dense, np.ndarray), f'Expected np matrix got {type(node_features_dense)}.'

    # The goal is to make feature vectors normalized (sum equals 1), but since some feature vectors are all 0s
    # in those cases we'd have division by 0 so I set the min value (via np.clip) to 1.
    # Note: 1 is a neutral element for division i.e. it won't modify the feature vector
    return node_features_dense / np.clip(node_features_dense.sum(1,keepdims=True), a_min=1, a_max=None)

def convert_adj_to_edge_index(adjacency_matrix):
    """
    """
    assert isinstance(adjacency_matrix, np.ndarray), f'Expected NumPy array got {type(adjacency_matrix)}.'
    height, width = adjacency_matrix.shape
    assert height == width, f'Expected square shape got = {adjacency_matrix.shape}.'

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    # active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] > 0:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)

def feature2adj(X_embed):
    edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(X_embed)
    graphdict = edgeList2edgeDict(edgeList, X_embed.shape[0])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = gae_util.mask_test_edges(adj)
    adj = adj_train

    return adj, adj_train, edgeList


def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """       

    edgeList=[]

    for i in np.arange(featureMatrix.shape[0]):
        tmp=featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0, res[0][1:k+1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0,res[0][j]]<=boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((i, res[0][j], weight))
    
    return edgeList

def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict