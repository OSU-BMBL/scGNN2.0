"""
"""
# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Main program for scGNN v2')
# Program related
parser.add_argument('--use_bulk', action='store_true', default=False, 
                    help='(boolean, default False) If True, expect a bulk expression file and will run deconvolution and imputation')
parser.add_argument('--given_cell_type_labels', action='store_true', default=False, 
                    help='(boolean, default False) If True, expect a cell type label file and will compute ARI against those labels')
parser.add_argument('--run_LTMG', action='store_true', default=False, 
                    help='(boolean, default False) Not fully implemented')
parser.add_argument('--use_CCC', action='store_true', default=False, 
                    help='(boolean, default False) Not fully implemented')
parser.add_argument('--dropout_prob', type=float, default=0.1, 
                    help='(float, default 0.1) Probability that a non-zero value in the sc expression matrix will be set to zero. If this is set to 0, will not perform dropout or compute imputation error ')
parser.add_argument('--seed', type=int, default=1, 
                    help='(int, default 1) Seed for torch and numpy random generators')
parser.add_argument('--total_epoch', type=int, default=31, 
                    help='(int, default 10) Total EM epochs')
parser.add_argument('--ari_threshold', type=float, default=0.95, 
                    help='(float, default 0.95) The threshold for ari')
parser.add_argument('--graph_change_threshold', type=float, default=0.01, 
                    help='(float, default 0.01) The threshold for graph change')
parser.add_argument('--alpha', type=float, default=0.5, 
                    help='(float, default 0.5)')


# Data loading related
parser.add_argument('--load_dataset_dir', type=str, default='/fs/ess/PCON0022/Edison/datasets', 
                    help='(str) Folder that stores all your datasets. For example, if your expression matrix is in /fs/ess/PCON1234/Brutus/datasets/12.Klein/T2000_expression.csv, this should be set to /fs/ess/PCON1234/Brutus/datasets')
parser.add_argument('--load_dataset_name', type=str, default='12.Klein', 
                    help='(str) Folder that contains all the relevant input files. For example, if your expression matrix is in /fs/ess/PCON1234/Brutus/datasets/12.Klein/T2000_expression.csv, this should be set to 12.Klein')
parser.add_argument('--load_use_benchmark', action='store_true', default=False, 
                    help='(boolean, default False) If True, expect the following files (replace DATASET_NAME with the input to the --load_dataset_name argument): `ind.DATASET_NAME.{x, tx, allx}`, `T2000_expression.csv`, `T2000_LTMG.txt`, `DATASET_NAME_cell_label.csv` if providing ground-truth cell type labels, and `DATASET_NAME_bulk.csv` if using bulk data')
parser.add_argument('--load_sc_dataset', type=str, default='', 
                    help='Not needed if using benchmark')
parser.add_argument('--load_bulk_dataset', type=str, default='', 
                    help='Not needed if using benchmark')
parser.add_argument('--load_cell_type_labels', type=str, default='', 
                    help='Not needed if using benchmark')
parser.add_argument('--load_LTMG', type=str, default=None, 
                    help='Not needed if using benchmark')

# Seurat related
parser.add_argument('--load_seurat_object', type=str, default=None, 
                    help='(str, default None) If not None, will load the csv generated from the SeuratObject specified in this file path')
# Rdata related
parser.add_argument('--load_rdata', type=str, default=None, 
                    help='(str, default None) rdata path')

# 10X related
parser.add_argument('--load_from_10X', type=str, default=None, 
                    help='(str, default None) If not None, will load the 10X data from this file path')


# Preprocess related
parser.add_argument('--preprocess_cell_cutoff', type=float, default=0.9, 
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_gene_cutoff', type=float, default=0.9, 
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_top_gene_select', type=int, default=2000, 
                    help='Not needed if using benchmark')

# Feature AE related
parser.add_argument('--feature_AE_epoch', nargs=2, type=int, default=[500, 300], 
                    help='(two integers separated by a space, default 500 200) First number being non-EM epochs, second number being EM epochs')
parser.add_argument('--feature_AE_batch_size', type=int, default=12800, 
                    help='(int, default 12800) Batch size')
parser.add_argument('--feature_AE_learning_rate', type=float, default=1e-3, 
                    help='(float, default 1e-3) Learning rate')
parser.add_argument('--feature_AE_regu_strength', type=float, default=0.9, 
                    help='(float, default 0.9) In loss function, this is the weight on the LTMG regularization matrix')
parser.add_argument('--feature_AE_dropout_prob', type=float, default=0, 
                    help='(float, default 0) The dropout probability for feature autoencoder')
parser.add_argument('--feature_AE_concat_prev_embed', type=str, default=None, 
                    help="(str, default None) Choose from {'feature', 'graph'}")               

# Graph AE related
parser.add_argument('--graph_AE_epoch', type=int, default=200,
                    help='(int, default 200) The epoch or graph autoencoder')
parser.add_argument('--graph_AE_use_GAT', action='store_true', default=False, 
                    help='(boolean, default False) If true, will use GAT for GAE layers; otherwise will use GCN layers')
parser.add_argument('--graph_AE_GAT_dropout', type=float, default=0,
                    help='(int, default 0) The dropout probability for GAT')
parser.add_argument('--graph_AE_learning_rate', type=float, default=1e-2, 
                    help='(float, default 1e-2) Learning rate')
parser.add_argument('--graph_AE_embedding_size', type=int, default=16, 
                    help='(int, default 16) Graphh AE embedding size')
parser.add_argument('--graph_AE_concat_prev_embed', action='store_true', default=False, 
                    help='(boolean, default False) If true, will concat GAE embed at t-1 with the inputed Feature AE embed at t for graph construction; else will construct graph using Feature AE embed only')
parser.add_argument('--graph_AE_normalize_embed', type=str, default=None, 
                    help="(str, default None) Choose from {None, 'sum1', 'binary'}")
parser.add_argument('--graph_AE_graph_construction', type=str, default='v2', 
                    help="(str, default v0) Choose from {'v0', 'v1', 'v2'}")
parser.add_argument('--graph_AE_neighborhood_factor', type=float, default=0.05,
                    help='(int, default 10)')
parser.add_argument('--graph_AE_retain_weights', action='store_true', default=False, 
                    help='(boolean, default False)')
parser.add_argument('--gat_multi_heads', type=int, default=2, 
                    help='(int, default 2)')                   
parser.add_argument('--gat_hid_embed', type=int, default=64, 
                    help='(int, default 64) The dim for hid_embed')   

# Clustering related
parser.add_argument('--clustering_louvain_only', action='store_true', default=False, 
                    help='(boolean, default False) If true, will use Louvain clustering only; otherwise, first use Louvain to determine clusters count (k), then perform KMeans.')
parser.add_argument('--clustering_use_flexible_k', action='store_true', default=False, 
                    help='(boolean, default False) If true, will determin k using Louvain every epoch; otherwise, will rely on the k in the first epoch')
parser.add_argument('--clustering_embed', type=str, default='graph', 
                    help="(str, default 'graph') Choose from {'feature', 'graph', 'both'}")
parser.add_argument('--clustering_method', type=str, default='KMeans', 
                    help="(str, default 'KMeans') Choose from {'KMeans', 'AffinityPropagation'}") 

# Cluster AE related
parser.add_argument('--cluster_AE_epoch', type=int, default=200,
                    help='(int, default 200) The epoch for cluster AE')
parser.add_argument('--cluster_AE_batch_size', type=int, default=12800, 
                    help='(int, default 12800) Batch size')
parser.add_argument('--cluster_AE_learning_rate', type=float, default=1e-3, 
                    help='(float, default 1e-3) Learning rate')
parser.add_argument('--cluster_AE_regu_strength', type=float, default=0.9, 
                    help='(float, default 0.9) In loss function, this is the weight on the LTMG regularization matrix')
parser.add_argument('--cluster_AE_dropout_prob', type=float, default=0, 
                    help='(float, default 0) The dropout probability for cluster AE')

# Deconvolution related
parser.add_argument('--deconv_opt1_learning_rate', type=float, default=1e-3, 
                    help='(float, default 1e-3) learning rate')
parser.add_argument('--deconv_opt1_epoch', type=int, default=5000, 
                    help='(int, default 5000) epoch')
parser.add_argument('--deconv_opt1_epsilon', type=float, default=1e-4, 
                    help='(float, default 1e-4) epsilon')
parser.add_argument('--deconv_opt1_regu_strength', type=float, default=1e-2, 
                    help='(flaot, default 1e-2) strength')

parser.add_argument('--deconv_opt2_learning_rate', type=float, default=1e-1, 
                    help='(flaot, default 1e-1) learning rate')
parser.add_argument('--deconv_opt2_epoch', type=int, default=500, 
                    help='(int, default 500) epoch')
parser.add_argument('--deconv_opt2_epsilon', type=float, default=1e-4, 
                    help='(flaot, default 1e-4) epsilon')
parser.add_argument('--deconv_opt2_regu_strength', type=float, default=1e-2, 
                    help='(float, default 1e-2) strength')

parser.add_argument('--deconv_opt3_learning_rate', type=float, default=1e-1, 
                    help='(float, default 1e-1)')
parser.add_argument('--deconv_opt3_epoch', type=int, default=150, 
                    help='(int, default 150) epoch')
parser.add_argument('--deconv_opt3_epsilon', type=float, default=1e-4, 
                    help='(flaot, default 1e-4) epsilon')
parser.add_argument('--deconv_opt3_regu_strength_1', type=float, default=0.8, 
                    help='(flaot, default 0.8) strength_1')
parser.add_argument('--deconv_opt3_regu_strength_2', type=float, default=1e-2, 
                    help='(flaot, default 1e-2) strength_2')
parser.add_argument('--deconv_opt3_regu_strength_3', type=float, default=1, 
                    help='(flaot, default 1) strength_3')

parser.add_argument('--deconv_tune_learning_rate', type=float, default=1e-2, 
                    help='(flaot, default 1e-2) learning rate')
parser.add_argument('--deconv_tune_epoch', type=int, default=20, 
                    help='(int, default 20) epoch')
parser.add_argument('--deconv_tune_epsilon', type=float, default=1e-4, 
                    help='(float, default) epsilon')

# Output related
parser.add_argument('--output_dir', type=str, default='outputs/', 
                    help="(str, default 'outputs/') Folder for storing all the outputs")
parser.add_argument('--output_run_ID', type=int, default=None, 
                    help='(int, default None) Run ID to be printed along with metric outputs')
parser.add_argument('--output_preprocessed', action='store_true', default=False, 
                    help='(boolean, default False) If true, will output preprocessed data and dropout info')
parser.add_argument('--output_intermediate', action='store_true', default=False, 
                    help='(boolean, default False) If true, will output intermediate results')
parser.add_argument('--output_rdata', action='store_true', default=False,
                    help='(boolean, default False) If true, will output rdata results')


args = parser.parse_args()

# Loading Packages
import info_log
info_log.print('\n> Loading Packages')
import torch
from time import time

# Local modules
import load
import preprocess
import benchmark_util
import util
# from ccs import CCC_graph_handler
import result
from auto_encoders.feature_AE import feature_AE_handler
from auto_encoders.graph_AE import graph_AE_handler
from auto_encoders.cluster_AE import cluster_AE_handler
from clustering import clustering_handler
from deconvolution import deconvolution_handler
from imputation import imputation_handler

# Set up the program
param = dict()
param['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param['dataloader_kwargs'] = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
param['tik'] = time()
torch.manual_seed(args.seed)
info_log.print( f"Using device: {param['device']}" )
info_log.print(args)

# Load and preprocess data
info_log.print('\n> Loading data ...')
X_sc_raw = load.sc_handler(args)
X_bulk_raw = load.bulk_handler(args) if args.use_bulk else None

info_log.print('\n> Preprocessing data ...')
X_sc = preprocess.sc_handler(X_sc_raw, args)
X_bulk = preprocess.bulk_handler(X_bulk_raw, gene_filter=X_sc['gene'])['expr'] if args.use_bulk else None

info_log.print('\n> Setting up data for testing ...')
x_dropout, dropout_info = benchmark_util.dropout(X_sc, args)
ct_labels_truth = load.cell_type_labels(args, cell_filter=X_sc['cell']) if args.given_cell_type_labels else None
result.write_out_preprocessed_data_for_benchmarking(X_sc, x_dropout, dropout_info, ct_labels_truth, args) if args.output_preprocessed else None

info_log.print('\n> Preparing other matrices ...')
if args.run_LTMG:
    from LTMG_R import runLTMG
    runLTMG(x_dropout, args)
TRS = load.LTMG_handler(X_sc, args) # cell * gene
CCC_graph = None # CCC_graph_handler(TRS, X_process) if args.use_CCC else None

# info_log.print('\n> Program Finished! \n')
# exit()

# Main program starts here
info_log.print('\n> Pre EM runs ...')
param['epoch_num'] = 0
param['total_epoch'] = args.total_epoch
x_dropout = x_dropout['expr']
param['n_feature_orig'] = x_dropout.shape[1]
param['x_dropout'] = x_dropout
X_process = x_dropout.copy()

X_embed, X_feature_recon, model_state = feature_AE_handler(X_process, TRS, args, param)

graph_embed, CCC_graph_hat, edgeList, adj = graph_AE_handler(X_embed, CCC_graph, args, param)

info_log.print('\n> Entering main loop ...')
metrics = result.Performance_Metrics(X_sc, X_process, X_feature_recon, edgeList, ct_labels_truth, dropout_info, graph_embed, X_embed, args, param)
for i in range(args.total_epoch):
    info_log.print(f"\n==========> scGNN Epoch {i+1}/{args.total_epoch} <==========")
    param['epoch_num'] = i+1
    param['feature_embed'] = X_embed
    param['graph_embed'] = graph_embed

    cluster_labels, cluster_lists_of_idx = clustering_handler(edgeList, args, param, metrics)

    param['impute_regu'] = util.graph_celltype_regu_handler(adj, cluster_labels)
    X_imputed_sc = cluster_AE_handler(X_feature_recon, TRS, cluster_lists_of_idx, args, param, model_state)

    if args.use_bulk:
        X_imputed_bulk = deconvolution_handler(X_process, X_bulk, x_dropout, TRS, cluster_lists_of_idx, args, param)
        X_imputed = imputation_handler(X_imputed_sc, X_imputed_bulk, x_dropout, args, param)
    else:
        X_imputed = X_imputed_sc
    
    X_embed, X_feature_recon, model_state = feature_AE_handler(X_imputed, TRS, args, param, model_state)

    graph_embed, CCC_graph_hat, edgeList, adj = graph_AE_handler(X_embed, CCC_graph, args, param)

    X_process = X_imputed

    # Evaluate performance metrics
    metrics.update(cluster_labels, X_imputed_sc, X_feature_recon, edgeList, graph_embed, X_embed, param)
    info_log.print(f"==========> Epoch {param['epoch_num']}: {metrics.latest_results()} <==========")

    if metrics.stopping_checks():
        info_log.print(f'\n> Converged! Early stopping')
        break

info_log.print('\n> Outputing results ...')
metrics.output(args)
result.write_out(X_sc, X_imputed_sc, cluster_labels, X_embed, graph_embed, edgeList, args, param)

# Plot & Print results
# info_log.print('\n> Plotting results ...')
# metrics.plot()
info_log.print(metrics.all_results())
info_log.print('\n> Program Finished! \n')
