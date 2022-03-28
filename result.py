import numpy as np
import pandas as pd

import os
import pickle as pkl
from time import time

import info_log
import util
import benchmark_util
from clustering import cluster_output_handler as cluster_summary
from deconvolution import average_by_ct

from sklearn.metrics import adjusted_rand_score, silhouette_score
import networkx as nx
from similarity_index_of_label_graph_package import similarity_index_of_label_graph_class


class Performance_Metrics():

    def __init__(self, X_sc, X_process, X_feature_recon, edgeList, ct_labels_truth, dropout_info, graph_embed, X_embed, args, param):
        
        # args
        self.given_cell_type_labels = args.given_cell_type_labels
        self.dropout_prob = args.dropout_prob
        self.total_epoch = args.total_epoch
        self.output_dir = args.output_dir
        self.ari_threshold = args.ari_threshold
        self.alpha = args.alpha
        self.use_bulk = args.use_bulk
        self.output_intermediate = args.output_intermediate

        # initialization
        self.cell = X_sc['cell']
        self.gene = X_sc['gene']
        self.ct_labels_truth = ct_labels_truth
        self.cluster_labels_old = np.zeros_like(X_sc['cell'])
        self.graph_old = nx.Graph()
        self.graph_old.add_weighted_edges_from(edgeList)
        self.adj_orig = nx.normalized_laplacian_matrix(self.graph_old)
        self.graph_change_threshold = args.graph_change_threshold * np.mean(abs(self.adj_orig))
        self.adjOld = self.adj_orig
        self.X_true = X_sc['expr']
        self.dropout_info = dropout_info
        self.graph_embed_old = graph_embed
        self.feature_embed_old = X_embed

        self.quick = True

        self.cluster_label_history = []

        # metric names and dictionary
        self.metric_names = [
            'ari_between_epochs', 'ari_against_ground_truth',
            'error_median', 'error_median_inv', 'error_median_entire',
            'deconv_error_median', 'deconv_error_median_inv', 'deconv_error_median_entire',
            'sc_error_median', 'sc_error_median_inv', 'sc_error_median_entire',
            'bulk_error_median', 'bulk_error_median_inv', 'bulk_error_median_entire',
            'sc_bulk_error_median', 'sc_bulk_error_median_inv', 'sc_bulk_error_median_entire',
            'feature_error_median', 'feature_error_median_inv', 'feature_error_median_entire',
            'error_median_by_ct',
            'graph_similarity', 'graph_change',
            'cluster_count', 'cluster_size_list',
            'silhouette_cluster', 'silhouette_embed',
            'silhouette_feature_embed', 'silhouette_graph_embed',
            'time_used'
        ]

        self.unused_metric_names = [] if not self.quick else ['silhouette_cluster', 'silhouette_embed',
            'silhouette_feature_embed', 'silhouette_graph_embed',
            'feature_error_median', 'feature_error_median_inv', 'feature_error_median_entire',
            'graph_similarity', 'graph_change']

        self.metrics = {name:[] for name in self.metric_names}

        # add initial values
        self.update(self.cluster_labels_old, X_process, X_feature_recon, edgeList, graph_embed, X_embed, param)
    
    def update(self, cluster_labels, X_imputed, X_feature_recon, edgeList, graph_embed, X_embed, param=None):
        
        info_log.print('--------> Computing all metrics ...')
        
        # Cluster info
        cluster_label_list = self.ct_labels_truth if param['epoch_num'] == 0 else cluster_labels
        cluster_index_list = cluster_summary(cluster_label_list)[1]
        cluster_count = len(cluster_index_list)
        cluster_size_list = [len(ct) for ct in cluster_index_list]
        cluster_size_list.sort()

        # Clustering evaluation
        ari_between_epochs = adjusted_rand_score(self.cluster_labels_old, cluster_labels) if param['epoch_num'] > 1 else None
        if param['epoch_num'] == 0:
            ari_against_ground_truth = None if self.given_cell_type_labels else self.unused_metric_names.append['ari_against_ground_truth']
        else:
            ari_against_ground_truth = adjusted_rand_score(self.ct_labels_truth, cluster_labels) 

        silhouette_cluster = silhouette_score(param['clustering_embed'], cluster_label_list) if not self.quick and param['epoch_num'] > 0 else None
        silhouette_embed = silhouette_score(param['clustering_embed'], self.ct_labels_truth) if not self.quick and param['epoch_num'] > 0 else None

        silhouette_feature_embed = silhouette_score(self.feature_embed_old, self.ct_labels_truth) if not self.quick else None
        silhouette_graph_embed = silhouette_score(self.graph_embed_old, self.ct_labels_truth) if not self.quick else None

        # Imputation evaluation
        name_list = [
                'deconv_error_median', 'deconv_error_median_inv', 'deconv_error_median_entire',
                'sc_error_median', 'sc_error_median_inv', 'sc_error_median_entire',
                'bulk_error_median', 'bulk_error_median_inv', 'bulk_error_median_entire',
                'sc_bulk_error_median', 'sc_bulk_error_median_inv', 'sc_bulk_error_median_entire',
                'error_median_by_ct'
            ]
            
        ## On imputed matrix
        error_median, error_median_inv, error_median_entire = benchmark_util.imputation_error_handler(X_imputed, self.X_true, self.dropout_prob, self.dropout_info)

        ## On Feature AE output
        feature_error_median, feature_error_median_inv, feature_error_median_entire = benchmark_util.imputation_error_handler(X_feature_recon, self.X_true, self.dropout_prob, self.dropout_info) if not self.quick else None, None, None

        # For Bulk Deconvolution testing
        if self.use_bulk and param['epoch_num'] == 0:

            # for name in name_list:
            #     exec(f'{name} = 0')
            
            deconv_error_median, deconv_error_median_inv, deconv_error_median_entire = None, None, None
            sc_error_median, sc_error_median_inv, sc_error_median_entire = None, None, None
            bulk_error_median, bulk_error_median_inv, bulk_error_median_entire = None, None, None
            sc_bulk_error_median, sc_bulk_error_median_inv, sc_bulk_error_median_entire = None, None, None
            error_median_by_ct = None

        elif self.use_bulk and param['epoch_num'] > 0:

            X_ct_avg = param['X_ct_avg'] # ct * gene
            X_deconvoluted_unadjusted = param['X_deconvoluted_unadjusted'] # cell * gene
            X_imputed_sc = param['X_imputed_sc'] # cell * gene
            X_imputed_bulk = param['X_imputed_bulk'] # cell * gene

            ## On unadjusted imputed bulk
            deconv_error_median, deconv_error_median_inv, deconv_error_median_entire = benchmark_util.imputation_error_handler(X_deconvoluted_unadjusted, self.X_true, self.dropout_prob, self.dropout_info)

            ## On imputed sc
            sc_error_median, sc_error_median_inv, sc_error_median_entire = benchmark_util.imputation_error_handler(X_imputed_sc, self.X_true, self.dropout_prob, self.dropout_info)

            ## On imputed bulk
            bulk_error_median, bulk_error_median_inv, bulk_error_median_entire = benchmark_util.imputation_error_handler(X_imputed_bulk, self.X_true, self.dropout_prob, self.dropout_info)

            ## By cell type
            X_orig_ct_avg_pos = average_by_ct(self.X_true.T, cluster_index_list, mask=self.X_true.T==0).T # ct * gene
            _, error_median_by_ct, _, _ = benchmark_util.imputation_error_entire(X_ct_avg, X_orig_ct_avg_pos)

            ## Between sc and bulk
            sc_bulk_error_median, sc_bulk_error_median_inv, sc_bulk_error_median_entire = benchmark_util.imputation_error_handler(X_imputed_sc, X_imputed_bulk, self.dropout_prob, self.dropout_info)
        
        elif param['epoch_num'] == 0:
            self.unused_metric_names.extend(name_list)
        
        if not self.dropout_prob and param['epoch_num'] == 0:
            self.unused_metric_names.extend(['error_median', 'error_median_inv'])

        # Graph similarity evaluation (beta)
        if not self.quick:
            graph_new = nx.Graph()
            graph_new.add_weighted_edges_from(edgeList)
            similarity_index_of_label_graph = similarity_index_of_label_graph_class()
            graph_similarity = similarity_index_of_label_graph(self.graph_old, graph_new) 

            # Graph changes
            adj_new_temp = nx.adjacency_matrix(graph_new)
            adjNew = self.alpha * self.adj_orig + (1- self.alpha) * adj_new_temp / np.sum(adj_new_temp, axis=0)
            graph_change = np.mean(abs(adjNew - self.adjOld))
        
        # Time elapsed
        tok = time()
        time_used = tok - param['tik'] # in seconds
        param['tik'] = tok

        # Log the latest metric values
        if param['epoch_num'] == 0:
            for name in self.unused_metric_names:
                self.metric_names.remove(name)
                self.metrics.pop(name)
                
        for name in self.metric_names:
            self.metrics[name].append(eval(name))
        
        self.cluster_label_history.append(cluster_label_list)

        self.output_intermediate_results(cluster_labels, graph_embed, X_embed, param) if self.output_intermediate else None

        # Update results for next iteration
        self.cluster_labels_old = cluster_labels
        self.graph_old = graph_new
        self.adjOld = adjNew
        self.graph_embed_old = graph_embed
        self.feature_embed_old = X_embed

    def output(self, args):
        info_log.print('--------> Exporting all metrics ...')
        
        run_ID = args.output_run_ID

        result_df = pd.DataFrame(data = self.metrics).T
        
        result_df = result_df.set_index([np.repeat(run_ID, len(result_df)), result_df.index]) if run_ID is not None else result_df

        result_df.to_csv(os.path.join(self.output_dir, 'all_metris.csv'))

        if args.output_intermediate:
            idx = ['Ground Truth']
            idx.extend([i+1 for i in range(len(self.cluster_label_history)-1)])
            ct_history = pd.DataFrame(self.cluster_label_history, index=idx, columns=self.cell).T
            ct_history.to_csv(os.path.join(self.output_dir, 'labels_history.csv'))

    def output_intermediate_results(self, cluster_labels, graph_embed, feature_embed, param, interval=5):
        output_dir = os.path.join(self.output_dir, 'intermediate')
        
        epoch_num = param['epoch_num']
        os.mkdir(output_dir) if epoch_num == 0 and not os.path.exists(output_dir) else None

        if epoch_num == 0 or epoch_num == 1 or epoch_num == param['total_epoch'] or epoch_num % interval ==0:

            info_log.print('--------> Exporting graph embeddings (intermediate) ...')
            embed_size = graph_embed.shape[1]
            emblist = []
            for i in range(embed_size):
                emblist.append(f'embedding_{i+1}')
            pd.DataFrame(data=graph_embed, index=self.cell, columns=emblist).to_csv(os.path.join(output_dir,f'graph_embedding_{epoch_num}.csv'))

            info_log.print('--------> Exporting feature embeddings (intermediate) ...')
            embed_size = feature_embed.shape[1]
            emblist = []
            for i in range(embed_size):
                emblist.append(f'embedding_{i+1}')
            pd.DataFrame(data=feature_embed, index=self.cell, columns=emblist).to_csv(os.path.join(output_dir,f'feature_embedding_{epoch_num}.csv'))

            if epoch_num > 0:
                info_log.print('--------> Exporting clustering embeddings (intermediate) ...')
                embed_size = param['clustering_embed'].shape[1]
                emblist = []
                for i in range(embed_size):
                    emblist.append(f'embedding_{i+1}')
                pd.DataFrame(data=param['clustering_embed'], index=self.cell, columns=emblist).to_csv(os.path.join(output_dir,f'clustering_embedding_{epoch_num}.csv'))
            
                util.drawUMAP(param['clustering_embed'], cluster_labels, output_dir, filename_suffix=f'pred_{epoch_num}')
                util.drawUMAP(param['clustering_embed'], self.ct_labels_truth, output_dir, filename_suffix=f'true_{epoch_num}')
                # util.drawTSNE(graph_embed, cluster_labels, output_dir)

    def plot(self):
        util.plot(self.metrics['ari_between_epochs'], ylabel='ARI Between Epochs', output_dir=self.output_dir)
        util.plot(self.metrics['cluster_silhouette_cluster'], ylabel='Silhouette of Clustering Labels', output_dir=self.output_dir)
        util.plot(self.metrics['cluster_silhouette_embed'], ylabel='Silhouette of Graph Embeddings', output_dir=self.output_dir)
        util.plot(self.metrics['ari_against_ground_truth'], ylabel='ARI Against Ground Truth', output_dir=self.output_dir) if self.given_cell_type_labels else None
        # util.plot(self.graph['similarity'], ylabel='Graph Similarity Between Epochs', output_dir=self.output_dir)
        util.plot(self.metrics['graph_change'], ylabel='Graph Change Between Epochs', hline=self.graph_change_threshold, output_dir=self.output_dir)
        util.plot(self.metrics['cluster_count'], ylabel='Cluster Count', output_dir=self.output_dir)
        if self.dropout_prob:
            util.plot(self.metrics['error_mean'], ylabel='Average L1 Error', output_dir=self.output_dir)
            util.plot(self.metrics['error_median'], ylabel='Median L1 Error', output_dir=self.output_dir)
            util.plot(self.metrics['error_min'], ylabel='Min L1 Error', output_dir=self.output_dir)
            util.plot(self.metrics['error_max'], ylabel='Max L1 Error', output_dir=self.output_dir)

    def latest_results(self):
        last_idx = len(self.metrics['ari_between_epochs']) - 1
        str_repr = f"ARI Between Epochs = {self.metrics['ari_between_epochs'][last_idx]}" + \
            (f", \nARI Against Ground Truth = {self.metrics['ari_against_ground_truth'][last_idx]}" if self.given_cell_type_labels else '') + \
            (f", \nMedian L1 Error = {self.metrics['error_median'][last_idx]}" if self.dropout_prob else '') + \
            f", \nPredicted {self.metrics['cluster_count'][last_idx]} clusters, their sizes are {self.metrics['cluster_size_list'][last_idx]}"
        return str_repr
    
    def all_results(self):
        
        ari_result_str = f"\n> ARI Against Ground Truth = {self.metrics['ari_against_ground_truth']}" if self.given_cell_type_labels else ''
        l1_result_str = f"\n> Median L1 Error = {self.metrics['error_median']}" if self.dropout_prob else ''
        total_time_str = f"\n> Total running time (seconds) = {sum(self.metrics['time_used'])}"

        return ari_result_str + l1_result_str + total_time_str

    def stopping_checks(self,):
        last_idx = len(self.metrics['ari_between_epochs']) - 1

        graph_change_is_small_enough = self.metrics['graph_change'][last_idx] < self.graph_change_threshold
        cell_type_pred_is_similar_enough = self.metrics['ari_against_ground_truth'][last_idx] > self.ari_threshold if self.given_cell_type_labels else False

        return graph_change_is_small_enough or cell_type_pred_is_similar_enough

def write_out(X_sc, X_imputed, cluster_labels, feature_embed, graph_embed, args, param):
    output_dir = args.output_dir

    info_log.print('--------> Exporting imputed expression matrix ...')
    pd.DataFrame(data=X_imputed, index=X_sc['cell'], columns=X_sc['gene']).to_csv(os.path.join(output_dir,'imputed.csv'))

    info_log.print('--------> Exporting cell label predictions ...')
    pd.DataFrame(data=cluster_labels, index=X_sc['cell'], columns=["labels_pred"]).to_csv(os.path.join(output_dir,'labels.csv'))

    info_log.print('--------> Exporting graph embeddings ...')
    embed_size = graph_embed.shape[1]
    emblist = []
    for i in range(embed_size):
        emblist.append(f'embedding_{i+1}')
    pd.DataFrame(data=graph_embed, index=X_sc['cell'], columns=emblist).to_csv(os.path.join(output_dir,'graph_embedding.csv'))

    info_log.print('--------> Exporting feature embeddings ...')
    embed_size = feature_embed.shape[1]
    emblist = []
    for i in range(embed_size):
        emblist.append(f'embedding_{i+1}')
    pd.DataFrame(data=feature_embed, index=X_sc['cell'], columns=emblist).to_csv(os.path.join(output_dir,'feature_embedding.csv'))

    info_log.print('--------> Exporting clustering embeddings ...')
    embed_size = param['clustering_embed'].shape[1]
    emblist = []
    for i in range(embed_size):
        emblist.append(f'embedding_{i+1}')
    pd.DataFrame(data=param['clustering_embed'], index=X_sc['cell'], columns=emblist).to_csv(os.path.join(output_dir,f'clustering_embedding.csv'))
    
    util.drawUMAP(param['clustering_embed'], cluster_labels, output_dir)
    util.drawTSNE(param['clustering_embed'], cluster_labels, output_dir)

def write_out_preprocessed_data_for_benchmarking(X_sc, x_dropout, dropout_info, ct_labels, args):
    output_dir = os.path.join(args.output_dir, 'preprocessed_data')
    dataset_name = args.load_dataset_name
    os.mkdir(output_dir) if not os.path.exists(output_dir) else None
    dropout_prob = args.dropout_prob

    if dropout_prob:
        info_log.print('--------> Exporting dropout data ...')
        pd.DataFrame(data=x_dropout['expr_b4_log'].T, index=x_dropout['gene'], columns=x_dropout['cell']).to_csv(os.path.join(output_dir,'dropout_top_expression.csv'))
    
        info_log.print('--------> Exporting dropout info ...')
        with open(os.path.join(output_dir,'dropout_info.pkl'), 'wb') as f:
            pkl.dump(dropout_info, f)
    
    info_log.print('--------> Exporting full data ...')
    pd.DataFrame(data=X_sc['expr_b4_log'].T, index=X_sc['gene'], columns=X_sc['cell']).to_csv(os.path.join(output_dir,'original_top_expression.csv'))

    # To compatible with scGNN 1.0 input interface
    feature = X_sc['expr_b4_log'] # cell * gene
    x = feature
    tx = feature[0:1]
    allx = feature[1:]

    pkl.dump(allx, open(os.path.join(output_dir,f"ind.{dataset_name}.allx"), "wb" ) )
    pkl.dump(x, open(os.path.join(output_dir,f"ind.{dataset_name}.x"), "wb" ) )
    pkl.dump(tx, open(os.path.join(output_dir,f"ind.{dataset_name}.tx"), "wb" ) )

    if args.given_cell_type_labels:
        info_log.print('--------> Exporting top cell labels ...')
        pd.DataFrame(data=ct_labels, index=X_sc['cell'], columns=['cell_type']).to_csv(os.path.join(output_dir,'top_cell_labels.csv')) 
