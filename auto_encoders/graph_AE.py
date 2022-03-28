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

import util
import auto_encoders.model as model
import info_log

def edgeList2edgeIndex(edgeList):
    result=[[i[0],i[1]] for i in edgeList]
    return result
    

def graph_AE_handler(X_embed, CCC_graph, args, param):
    info_log.print('--------> Starting Graph AE ...')

    use_GAT = args.graph_AE_use_GAT
    learning_rate = args.graph_AE_learning_rate
    total_epoch = args.graph_AE_epoch
    embedding_size = args.graph_AE_embedding_size
    concat_prev_embed = args.graph_AE_concat_prev_embed
    normalize_embed = args.graph_AE_normalize_embed
    gat_dropout = args.graph_AE_GAT_dropout
    graph_construction = args.graph_AE_graph_construction
    neighborhood_factor = args.graph_AE_neighborhood_factor
    retain_weights = args.graph_AE_retain_weights

    if concat_prev_embed and param['epoch_num'] > 0:
        graph_embed = param['graph_embed']
        graph_embed_norm = util.normalizer(graph_embed, base=X_embed, axis=0)
        X_embed = np.concatenate((X_embed, graph_embed_norm), axis=1)

    
    if normalize_embed == 'sum1':
        zDiscret = normalize_features_dense(X_embed)
    elif normalize_embed == 'binary':
        zDiscret = int(X_embed > np.mean(X_embed, axis=0))
    else:
        zDiscret = X_embed

    adj, adj_train, edgeList = feature2adj(X_embed, graph_construction, neighborhood_factor, retain_weights)
    adj_norm = gae_util.preprocess_graph(adj_train)
    adj_label = (adj_train + sp.eye(adj_train.shape[0])).toarray()

    # Prepare matrices
    if use_GAT:
        edgeIndex=edgeList2edgeIndex(edgeList)
        edgeIndex=np.array(edgeIndex).T
        CCC_graph_edge_index = torch.from_numpy(edgeIndex).type(torch.LongTensor).to(param['device'])
    else:
        CCC_graph_edge_index = adj_norm.to(param['device']) 

        pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
        norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
    
    X_embed_normalized = torch.from_numpy(zDiscret).type(torch.FloatTensor).to(param['device'])
    CCC_graph = torch.from_numpy(adj_label).type(torch.FloatTensor).to(param['device'])

    graph_AE = model.Graph_AE(X_embed.shape[1], embedding_size, gat_dropout).to(param['device'])
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

def feature2adj(X_embed, graph_construction, neighborhood_factor, retain_weights):
    neighborhood_size_temp = neighborhood_factor if neighborhood_factor > 1 else round(X_embed.shape[0] * neighborhood_factor)
    neighborhood_size = neighborhood_size_temp - 1 if neighborhood_size_temp == X_embed.shape[0] else neighborhood_size_temp

    if graph_construction == 'v0':
        edgeList = v0_calculateKNNgraphDistanceMatrixStatsSingleThread(X_embed)
    elif graph_construction == 'v1':
        edgeList = v1_calculateKNNgraphDistanceMatrixStatsSingleThread(X_embed, k=neighborhood_size)
    elif graph_construction == 'v2':
        edgeList = v2_calculateKNNgraphDistanceMatrixStatsSingleThread(X_embed, k=neighborhood_size)
    
    if retain_weights:
        G = nx.DiGraph()
        G.add_weighted_edges_from(edgeList)
        adj_return = nx.adjacency_matrix(G)
    else:
        graphdict = edgeList2edgeDict(edgeList, X_embed.shape[0])
        adj_return = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    adj = adj_return.copy()
    # adj_orig = adj

    # Clear diagonal elements (no self loop)
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()

    # build test set with 10% positive links, edge lists only contain single direction of edge!
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = gae_util.mask_test_edges(adj)

    # Clear diagonal elements (no self loop)
    adj_train = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_train.eliminate_zeros()

    return adj_return, adj_train, edgeList


def test_calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10, verbose=True):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """       

    edgeList=[]

    for i in np.arange(featureMatrix.shape[0]):
        print(f'\ni:{i}') if verbose else None
        tmp=featureMatrix[i,:].reshape(1,-1)
        print(f'tmp:{tmp}') if verbose else None
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        print(f'distMat:{distMat}') if verbose else None
        res = distMat.argsort()[:k+1]
        print(f'res:{res}') if verbose else None
        tmpdist = distMat[0, res[0][1:k+1]]
        print(f'tmpdist:{tmpdist}') if verbose else None
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        print(f'boundary:{boundary}') if verbose else None
        for j in np.arange(1, k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            # if distMat[0,res[0][j]]<=boundary:
            #     weight = 1.0-distMat[0,res[0][j]]/boundary#1.0
            # else:
            #     weight = 0.0
            weight = -distMat[0,res[0][j]]
            print(f'    {j} weight:{weight}') if verbose else None
            edgeList.append((i, res[0][j], weight))
    
    return edgeList

def v0_calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10):
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

def v1_calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """       
    edgeList=[]
    distance_dist=[]

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
                weight = 1.0-distMat[0,res[0][j]]/boundary
                distance_dist.append(weight)
            else:
                weight = 0.0
            edgeList.append((i, res[0][j], weight))

    # with open("/users/PCON0022/haocheng/scGNN/scGNN2.0/weights.txt",'a+') as f:
    #     list_str=",".join(map(str,distance_dist))
    #     f.write(f'{list_str}\n')

    return edgeList

def v2_calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """       

    edgeList=[]

    for i in np.arange(featureMatrix.shape[0]):
        tmp=featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0, res[0][1:k+1]]
        # boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k+1):
            weight = 1 / (distMat[0,res[0][j]] + 1e-16)
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