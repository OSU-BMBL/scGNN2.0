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
parser.add_argument('--total_epoch', type=int, default=10, 
                    help='(int, default 10) Total EM epochs')
parser.add_argument('--seed', type=int, default=1, 
                    help='(int, default 1) Seed for torch and numpy random generators')

# Data loading related
parser.add_argument('--load_dataset_dir', type=str, default='/fs/ess/PCON0022/Edison/datasets', 
                    help='(str) Folder that stores all your datasets. For example, if your expression matrix is in /fs/ess/PCON1234/Brutus/datasets/12.Klein/T2000_expression.csv, this should be set to /fs/ess/PCON1234/Brutus/datasets')
parser.add_argument('--load_dataset_name', type=str, default='12.Klein', 
                    help='(str) Folder that contains all the relevant input files. For example, if your expression matrix is in /fs/ess/PCON1234/Brutus/datasets/12.Klein/T2000_expression.csv, this should be set to 12.Klein')
parser.add_argument('--load_use_benchmark', action='store_true', default=False, 
                    help='(boolean, default False) If True, expect the following files (replace DATASET_NAME with the input to the --load_dataset_name argument): `ind.DATASET_NAME.{x, tx, allx}`, `T2000_expression.csv`, `T2000_LTMG.txt`, `DATASET_NAME__cell_label.csv` if providing ground-truth cell type labels, and `DATASET_NAME_bulk.csv` if using bulk data')
parser.add_argument('--load_sc_dataset', type=str, default='', 
                    help='Not needed if using benchmark')
parser.add_argument('--load_bulk_dataset', type=str, default='', 
                    help='Not needed if using benchmark')
parser.add_argument('--load_cell_type_labels', type=str, default='', 
                    help='Not needed if using benchmark')
parser.add_argument('--load_LTMG', type=str, default='', 
                    help='Not needed if using benchmark')

# Preprocess related
parser.add_argument('--preprocess_cell_cutoff', type=float, default=0.9, 
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_gene_cutoff', type=float, default=0.9, 
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_top_gene_select', type=int, default=2000, 
                    help='Not needed if using benchmark')

# Feature AE related
parser.add_argument('--feature_AE_epoch', nargs=2, type=int, default=[500, 200], 
                    help='(two integers separated by a space, default 500 200) First number being non-EM epochs, second number being EM epochs')
parser.add_argument('--feature_AE_batch_size', type=int, default=12800, 
                    help='(int, default 12800) Batch size')
parser.add_argument('--feature_AE_learning_rate', type=float, default=1e-3, 
                    help='(float, default 1e-3) Learning rate')
parser.add_argument('--feature_AE_regu_strength', type=float, default=0.9, 
                    help='(float, default 0.9) In loss function, this is the weight on the LTMG regularization matrix')

# Graph AE related
parser.add_argument('--graph_AE_epoch', type=int, default=200,
                    help='(int, default 200)')
parser.add_argument('--graph_AE_use_GAT', action='store_true', default=False, 
                    help='(boolean, default False) Not fully implemented')
parser.add_argument('--graph_AE_learning_rate', type=float, default=1e-2, 
                    help='(float, default 1e-2) Learning rate')

# Cluster AE related
parser.add_argument('--cluster_AE_epoch', type=int, default=200,
                    help='(int, default 200)')
parser.add_argument('--cluster_AE_batch_size', type=int, default=12800, 
                    help='(int, default 12800) Batch size')
parser.add_argument('--cluster_AE_learning_rate', type=float, default=1e-3, 
                    help='(float, default 1e-3) Learning rate')
parser.add_argument('--cluster_AE_regu_strength', type=float, default=0.9, 
                    help='(float, default 0.9) In loss function, this is the weight on the LTMG regularization matrix')

# Deconvolution related
parser.add_argument('--deconv_opt1_learning_rate', type=float, default=1e-3, 
                    help='')
parser.add_argument('--deconv_opt1_epoch', type=int, default=5000, 
                    help='')
parser.add_argument('--deconv_opt1_epsilon', type=float, default=1e-4, 
                    help='')
parser.add_argument('--deconv_opt1_regu_strength', type=float, default=1e-2, 
                    help='')

parser.add_argument('--deconv_opt2_learning_rate', type=float, default=1e-1, 
                    help='')
parser.add_argument('--deconv_opt2_epoch', type=int, default=500, 
                    help='')
parser.add_argument('--deconv_opt2_epsilon', type=float, default=1e-4, 
                    help='')
parser.add_argument('--deconv_opt2_regu_strength', type=float, default=1e-2, 
                    help='')

parser.add_argument('--deconv_opt3_learning_rate', type=float, default=1e-2, 
                    help='')
parser.add_argument('--deconv_opt3_epoch', type=int, default=20, 
                    help='')
parser.add_argument('--deconv_opt3_epsilon', type=float, default=1e-4, 
                    help='')
parser.add_argument('--deconv_opt3_regu_strength_1', type=float, default=1e-2, 
                    help='')
parser.add_argument('--deconv_opt3_regu_strength_2', type=float, default=1e-2, 
                    help='')

parser.add_argument('--deconv_tune_learning_rate', type=float, default=1e-2, 
                    help='')
parser.add_argument('--deconv_tune_epoch', type=int, default=20, 
                    help='')
parser.add_argument('--deconv_tune_epsilon', type=float, default=1e-4, 
                    help='')

# Output related
parser.add_argument('--output_dir', type=str, default='outputs/', 
                    help='(str) Folder for storing all the outputs')

args = parser.parse_args()

# Loading Packages
import info_log
info_log.print('\n> Loading Packages')
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Local modules
import load
import preprocess
import benchmark_util
# from LTMG_R import LTMG_handler # commented out for benchmark testing
# from ccs import CCC_graph_handler
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
torch.manual_seed(args.seed)
info_log.print( f"Using device: {param['device']}" )
info_log.print(args)

# Loading and preprocessing data
info_log.print('\n> Loading data ...')
X_sc_raw = load.sc_handler(args)
X_bulk_raw = load.bulk_handler(args) if args.use_bulk else None

info_log.print('\n> Preprocessing data ...')
X_sc = preprocess.sc_handler(X_sc_raw, args)
X_bulk = preprocess.bulk_handler(X_bulk_raw, gene_filter=X_sc['gene'])['expr'] if args.use_bulk else None

info_log.print('\n> Setting up data for testing ...')
if args.dropout_prob:
    X_process, dropout_info = benchmark_util.dropout(X_sc['expr'], args)
else:
    X_process, dropout_info = X_sc['expr'].copy(), None
ct_labels_truth = load.cell_type_labels(args, cell_filter=X_sc['cell']) if args.given_cell_type_labels else None

info_log.print('\n> Preparing other matrices ...')
TRS = load.LTMG_handler(args)
CCC_graph = None # CCC_graph_handler(TRS, X_process) if args.use_CCC else None

info_log.print('\n> Pre EM runs ...')
param['is_EM'] = False
X_embed, _, model_state = feature_AE_handler(X_process, TRS, args, param)
graph_embed, CCC_graph_hat, edgeList, adj = graph_AE_handler(X_embed, CCC_graph, args, param)

info_log.print('\n> Entering main loop')
cluster_labels_old = [1 for i in range(len(X_sc['cell']))]
ARI_between_epochs_history, ARI_against_ground_truth_history, avg_err_history, med_err_history, min_err_history, max_err_history = [], [], [], [], [], []
# cluster_AE_state_dict = deepcopy(feature_AE_state_dict)
for i in range(args.total_epoch):
    info_log.print(f"\n==========> scGNN Epoch {i+1}/{args.total_epoch} <==========")

    param['is_EM'] = True
    cluster_labels, cluster_lists_of_idx = clustering_handler(graph_embed, edgeList) # Changed from X_embed to CCC_graph_hat (test both), now to edgeList just for test benchmarking
    X_imputed_sc = cluster_AE_handler(X_process, TRS, cluster_lists_of_idx, args, param, model_state)

    if args.use_bulk:
        X_imputed_bulk = deconvolution_handler(X_process, X_bulk, TRS, cluster_lists_of_idx, args, param)
        X_imputed = imputation_handler(X_imputed_sc, X_imputed_bulk, X_process, i+1, args.output_dir)
    else:
        X_imputed = X_imputed_sc

    # feature_AE_state_dict = deepcopy(cluster_AE_state_dict)
    X_embed, _, model_state = feature_AE_handler(X_imputed, TRS, args, param, model_state) # X_imputed_sc and X_recon is cell * gene, should not use X_process['expr']
    graph_embed, CCC_graph_hat, edgeList, adj = graph_AE_handler(X_embed, CCC_graph, args, param)

    # X_imputed = X_recon
    X_process = X_imputed

    # Code below will be cleaned up in the next update
    ##########
    ARI_between_epochs = adjusted_rand_score(cluster_labels_old, cluster_labels)
    ARI_against_ground_truth = adjusted_rand_score(ct_labels_truth, cluster_labels) if args.given_cell_type_labels else None
    avg_err, med_err, min_err, max_err = None, None, None, None
    if args.dropout_prob:
        avg_err, med_err, min_err, max_err = benchmark_util.imputation_error(X_imputed, X_sc['expr'], *dropout_info)
    ##########  
    ARI_between_epochs_history.append(ARI_between_epochs)
    ARI_against_ground_truth_history.append(ARI_against_ground_truth)
    avg_err_history.append(avg_err)
    med_err_history.append(med_err)
    min_err_history.append(min_err)
    max_err_history.append(max_err)
    ##########
    if args.given_cell_type_labels:
        plt.plot(range(i+1), ARI_against_ground_truth_history)
        plt.xlabel('epochs')
        plt.ylabel('ARI Against Ground Truth')
        plt.savefig(os.path.join(args.output_dir, "ARI_against_ground_truth.png"))
        plt.clf()
    ##########
    if args.dropout_prob:
        plt.plot(range(i+1), med_err_history)
        plt.xlabel('epochs')
        plt.ylabel('Median L1 Error per epoch')
        plt.savefig(os.path.join(args.output_dir, "med_err.png"))
        plt.clf()
    ##########
    info_log.print(f"==========> Epoch {i+1}: ARI Between Epochs = {ARI_between_epochs}" +
          (f", ARI Against Ground Truth = {ARI_against_ground_truth}" if args.given_cell_type_labels else '') +
          (f", Median L1 Error = {med_err}" if args.dropout_prob else '') +
          " <==========")

    # Update
    # X_process['expr'] = X_imputed
    cluster_labels_old = cluster_labels

info_log.print('\n> Plotting results')

plt.plot(range(args.total_epoch), ARI_between_epochs_history)
plt.xlabel('epochs')
plt.ylabel('ARI Between Epochs')
plt.savefig(os.path.join(args.output_dir, "ARI_between_epochs.png"))
plt.clf()

if args.given_cell_type_labels:
    plt.plot(range(args.total_epoch), ARI_against_ground_truth_history)
    plt.xlabel('epochs')
    plt.ylabel('ARI Against Ground Truth')
    plt.savefig(os.path.join(args.output_dir, "ARI_against_ground_truth.png"))
    plt.clf()

if args.dropout_prob:
    plt.plot(range(args.total_epoch), avg_err_history)
    plt.xlabel('epochs')
    plt.ylabel('Average L1 Error per epoch')
    plt.savefig(os.path.join(args.output_dir, "avg_err.png"))
    plt.clf()

    plt.plot(range(args.total_epoch), med_err_history)
    plt.xlabel('epochs')
    plt.ylabel('Median L1 Error per epoch')
    plt.savefig(os.path.join(args.output_dir, "med_err.png"))
    plt.clf()

    plt.plot(range(args.total_epoch), min_err_history)
    plt.xlabel('epochs')
    plt.ylabel('Min L1 Error per epoch')
    plt.savefig(os.path.join(args.output_dir, "min_err.png"))
    plt.clf()

    plt.plot(range(args.total_epoch), max_err_history)
    plt.xlabel('epochs')
    plt.ylabel('Max L1 Error per epoch')
    plt.savefig(os.path.join(args.output_dir, "max_err.png"))
    plt.clf()

info_log.print(f'\n> ARI Against Ground Truth History: {ARI_against_ground_truth_history}') if args.given_cell_type_labels else None
info_log.print(f'\n> Median L1 Error History: {med_err_history}') if args.dropout_prob else None

info_log.print('\n> Program Finished! \n')
