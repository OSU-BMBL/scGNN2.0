import numpy as np
import pandas as pd

import os
import pickle as pkl

import info_log
import util
import benchmark_util
from clustering import cluster_output_handler as cluster_summary

from sklearn.metrics import adjusted_rand_score
import networkx as nx
# from similarity_index_of_label_graph_package import similarity_index_of_label_graph_class


class Performance_Metrics():

    def __init__(self, X_sc, X_process, edgeList, ct_labels_truth, dropout_info, args):

        self.given_cell_type_labels = args.given_cell_type_labels
        self.dropout_prob = args.dropout_prob
        self.total_epoch = args.total_epoch
        self.output_dir = args.output_dir
        self.ari_threshold = args.ari_threshold
        self.alpha = args.alpha

        self.ct_labels_truth = ct_labels_truth
        self.cluster_labels_old = np.zeros_like(X_sc['cell'])
        self.graph_old = nx.Graph()
        self.graph_old.add_weighted_edges_from(edgeList)
        self.adj_orig = nx.normalized_laplacian_matrix(self.graph_old)
        self.graph_change_threshold = args.graph_change_threshold * np.mean(abs(self.adj_orig))
        self.adjOld = self.adj_orig
        self.X_true = X_sc['expr']
        self.dropout_info = dropout_info

        self.ARI = {
            'between_epochs': [],
            'against_ground_truth': []
        }

        self.error = {
            'mean': [],
            'median': [],
            'min': [],
            'max': []
        }

        self.graph = {
            'similarity': [],
            'change': []
        }

        self.cluster = {
            'count': [],
            'size_list': []
        }

        self.update(self.cluster_labels_old+1, X_process, edgeList)
    
    def update(self, cluster_labels, X_imputed, edgeList):

        # Clustering evaluation
        ARI_between_epochs = adjusted_rand_score(self.cluster_labels_old, cluster_labels)
        ARI_against_ground_truth = adjusted_rand_score(self.ct_labels_truth, cluster_labels) if self.given_cell_type_labels else None

        # Imputation evaluation
        avg_err, med_err, min_err, max_err = None, None, None, None
        if self.dropout_prob:
            avg_err, med_err, min_err, max_err = benchmark_util.imputation_error(X_imputed, self.X_true, *self.dropout_info)

        # Graph similarity evaluation (beta)
        graph_new = nx.Graph()
        graph_new.add_weighted_edges_from(edgeList)
        # similarity_index_of_label_graph = similarity_index_of_label_graph_class()
        # graph_similarity_between_epochs = similarity_index_of_label_graph(self.graph_old, graph_new)

        # Graph changes
        adj_new_temp = nx.adjacency_matrix(graph_new)
        adjNew = self.alpha * self.adj_orig + (1- self.alpha) * adj_new_temp / np.sum(adj_new_temp, axis=0)
        graphChange = np.mean(abs(adjNew - self.adjOld))

        # Cluster info
        cluster_label_list = self.ct_labels_truth if len(self.cluster['count']) == 0 else cluster_labels
        cluster_index_list = cluster_summary(cluster_label_list)[1]
        cluster_count = len(cluster_index_list)
        cluster_size_list = [len(ct) for ct in cluster_index_list]
        
        # Log the latest metric values
        self.ARI['between_epochs'].append(ARI_between_epochs)
        self.ARI['against_ground_truth'].append(ARI_against_ground_truth)
        self.error['mean'].append(avg_err)
        self.error['median'].append(med_err)
        self.error['min'].append(min_err)
        self.error['max'].append(max_err)
        # self.graph['similarity'].append(graph_similarity_between_epochs)
        self.graph['change'].append(graphChange)
        self.cluster['count'].append(cluster_count)
        self.cluster['size_list'].append(cluster_size_list)

        # Update results for next iteration
        self.cluster_labels_old = cluster_labels
        self.graph_old = graph_new
        self.adjOld = adjNew

    def output(self):
        info_log.print('--------> Exporting all metrics ...')
        
        result_df = pd.DataFrame(
            data = {
                'ari_against_ground_truth': self.ARI['against_ground_truth'],
                'error_median': self.error['median'],
                'ari_between_epochs': self.ARI['between_epochs'],
                'error_mean': self.error['mean'],
                'error_min': self.error['min'],
                'error_max': self.error['max'],
                'graph_change': self.graph['change'],
                'cluster_count': self.cluster['count'],
                'cluster_size_list': self.cluster['size_list']
            }
        )
        result_df.to_csv(os.path.join(self.output_dir, 'all_metris.csv'))

    def plot(self):
        util.plot(self.ARI['between_epochs'], ylabel='ARI Between Epochs', output_dir=self.output_dir)
        util.plot(self.ARI['against_ground_truth'], ylabel='ARI Against Ground Truth', output_dir=self.output_dir) if self.given_cell_type_labels else None
        # util.plot(self.graph['similarity'], ylabel='Graph Similarity Between Epochs', output_dir=self.output_dir)
        util.plot(self.graph['change'], ylabel='Graph Change Between Epochs', hline=self.graph_change_threshold, output_dir=self.output_dir)
        util.plot(self.cluster['count'], ylabel='Cluster Count', output_dir=self.output_dir)
        if self.dropout_prob:
            util.plot(self.error['mean'], ylabel='Average L1 Error', output_dir=self.output_dir)
            util.plot(self.error['median'], ylabel='Median L1 Error', output_dir=self.output_dir)
            util.plot(self.error['min'], ylabel='Min L1 Error', output_dir=self.output_dir)
            util.plot(self.error['max'], ylabel='Max L1 Error', output_dir=self.output_dir)

    def latest_results(self):
        last_idx = len(self.ARI['between_epochs']) - 1
        str_repr = f"ARI Between Epochs = {self.ARI['between_epochs'][last_idx]}" + \
            (f", \nARI Against Ground Truth = {self.ARI['against_ground_truth'][last_idx]}" if self.given_cell_type_labels else '') + \
            (f", \nMedian L1 Error = {self.error['median'][last_idx]}" if self.dropout_prob else '') + \
            f", \nPredicted {self.cluster['count'][last_idx]} clusters, their sizes are {self.cluster['size_list'][last_idx]}"
        return str_repr
    
    def all_results(self):
        str_repr = (f"\n> ARI Against Ground Truth = {self.ARI['against_ground_truth']}" if self.given_cell_type_labels else '') + \
            (f"\n> Median L1 Error = {self.error['median']}" if self.dropout_prob else '')
        return str_repr

    def stopping_checks(self,):
        last_idx = len(self.ARI['between_epochs']) - 1
        
        return self.graph['change'][last_idx] < self.graph_change_threshold or (self.ARI['against_ground_truth'][last_idx] > self.ari_threshold if self.given_cell_type_labels else False)

def write_out(X_sc, cluster_labels, graph_embed, args):
    output_dir = args.output_dir

    info_log.print('--------> Exporting cell label predictions ...')
    pd.DataFrame(data=cluster_labels, index=X_sc['cell'], columns=["labels_pred"]).to_csv(os.path.join(output_dir,'labels.csv'))

    info_log.print('--------> Exporting graph embeddings ...')
    emblist = []
    for i in range(args.graph_AE_embedding_size):
        emblist.append(f'embedding_{i+1}')
    pd.DataFrame(data=graph_embed, index=X_sc['cell'], columns=emblist).to_csv(os.path.join(output_dir,'embedding.csv'))
    
def write_out_dropout_data(X_sc, X_process, dropout_info, args):
    output_dir = args.output_dir

    info_log.print('--------> Exporting dropout data ...')
    pd.DataFrame(data=X_process.T, index=X_sc['gene'], columns=X_sc['cell']).to_csv(os.path.join(output_dir,'dropout_expression.csv'))
    
    info_log.print('--------> Exporting dropout info ...')
    with open(os.path.join(output_dir,'dropout_info.pkl'), 'wb') as f:
        pkl.dump(dropout_info, f)
    
    info_log.print('--------> Exporting full data ...')
    pd.DataFrame(data=X_sc['expr'].T, index=X_sc['gene'], columns=X_sc['cell']).to_csv(os.path.join(output_dir,'original_expression.csv'))
    
    info_log.print('Program finished')
    exit()
    
    