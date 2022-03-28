"""
"""
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation

import random
import networkx as nx
from igraph import * # ignore the squiggly underline, not an error

import util
import info_log

def clustering_handler(edgeList, args, param, metrics):
    info_log.print('--------> Start Clustering ...')

    random.seed(args.seed)
    np.random.seed(args.seed)

    louvain_only = args.clustering_louvain_only
    use_flexible_k = args.clustering_use_flexible_k
    all_ct_count = metrics.metrics['cluster_count']
    clustering_embed = args.clustering_embed
    clustering_method = args.clustering_method
    avg_factor = 0.99

    if clustering_embed == 'graph':
        embed = param['graph_embed']
    elif clustering_embed == 'feature':
        embed = param['feature_embed']
    elif clustering_embed == 'both':
        feature_embed_norm = util.normalizer(param['feature_embed'], base=param['graph_embed'], axis=0)
        embed = np.concatenate((param['graph_embed'], feature_embed_norm), axis=1)
    else:
        info_log.print('--------> clustering_embed argument not recognized, using graph embed ...')
        embed = param['graph_embed']
    
    param['clustering_embed'] = embed

    listResult, size = generateLouvainCluster(edgeList)  # edgeList = (cell_i, cell_a), (cell_i, cell_b), ...
    k_Louvain = len(np.unique(listResult))
    info_log.print(f'----------------> Louvain clusters count: {k_Louvain}')

    resolution =  0.8 if embed.shape[0] < 2000 else 0.5 # based on num of cells
    k_resolution = k_Louvain * resolution

    if use_flexible_k or len(all_ct_count) == 1:
        k = max(round(k_resolution), 2)
    else:
        k_prev = all_ct_count[-1]
        k_exp = avg_factor * k_prev + (1-avg_factor) * k_resolution
        k = max(round(k_exp), 2)
        info_log.print(f'----------------> Exponentially adjusted clusters count: {k}')
        k = k_prev
    
    if not louvain_only:
        # resolution =  0.8 if embed.shape[0] < 2000 else 0.5 # based on num of cells
        # k = int(k * resolution) if int(k * resolution) >= 3 else 2

        if clustering_method == 'KMeans':
            listResult = KMeans(n_clusters=k, random_state=0).fit_predict(embed)  # (n_samples,) Index of the cluster each sample belongs to
        elif clustering_method == 'AffinityPropagation':
            listResult = AffinityPropagation(random_state=args.seed).fit_predict(embed)

    if len(set(listResult)) > 30 or len(set(listResult)) <= 1:
        info_log.print(f"----------------> Stopping: Number of clusters is {len(set(listResult))}")
        listResult = trimClustering(
            listResult, minMemberinCluster=5, maxClusterNumber=30)

    info_log.print(f'----------------> Total Cluster Number: {len(set(listResult))}')
    return cluster_output_handler(listResult) # tuple{'ct_list', 'lists_of_idx'}

def generateLouvainCluster(edgeList):
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode='undirected', attr="weight", loops=False) # ignore the squiggly underline, not errors

    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size

def convert_adj_to_edge_list(adjacency_matrix):
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
                edge_index.append((src_node_id, trg_nod_id, adjacency_matrix[src_node_id, trg_nod_id]))

    return np.asarray(edge_index)  # (N,3)

def cluster_output_handler(listResult):
    clusterIndexList = []
    for i in range(len(set(listResult))):
        clusterIndexList.append([])
    for i in range(len(listResult)):
        clusterIndexList[listResult[i]].append(i)

    return listResult, clusterIndexList
    
    # {
    #     'ct_list': listResult,
    #     'lists_of_idx' : clusterIndexList
    # }

def trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30):
    '''
    If the clustering numbers larger than certain number, use this function to trim. May have better solution
    '''
    numDict = {}
    for item in listResult:
        if not item in numDict:
            numDict[item] = 0
        else:
            numDict[item] = numDict[item]+1

    size = len(set(listResult))
    changeDict = {}
    for item in range(size):
        if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
            changeDict[item] = ''

    count = 0
    for item in listResult:
        if item in changeDict:
            listResult[count] = maxClusterNumber
        count += 1

    return listResult