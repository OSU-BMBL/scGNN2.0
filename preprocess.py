"""
    1) Load
    2) Truncate
    3) Sort and select
    4) Take natural log

    conda activate /Users/edisongu/opt/anaconda3/envs/scgnnEnv

"""

import numpy as np

import info_log

def sc_handler(X_raw, args):
    info_log.print('--------> Preprocessing sc data ...')

    cell_cutoff = args.preprocess_cell_cutoff
    gene_cutoff = args.preprocess_gene_cutoff
    num_gene_select = args.preprocess_top_gene_select

    if args.load_use_benchmark:
        X_select = X_raw
    else:
        X_trunc = filter(X_raw, cell_cutoff=cell_cutoff, gene_cutoff=gene_cutoff)
        X_select = sort(X_trunc, num_select=num_gene_select)

    X_log = log_transform(X_select)

    info_log.print(f"--------> Preprocessed sc data has {len(X_log['cell'])} cells and {len(X_log['gene'])} genes, Removing {len(X_raw['cell']) - len(X_log['cell'])} cells and {len(X_raw['gene']) - len(X_log['gene'])} genes")
    
    return X_log # cell * gene

def bulk_handler(X_raw, gene_filter):
    info_log.print('--------> Preprocessing bulk data ...')

    X_trunc = filter(X_raw, gene_filter=gene_filter)
    X_log = log_transform(X_trunc)

    info_log.print(f"--------> Preprocessed bulk data has {len(X_log['cell'])} cells and {len(X_log['gene'])} genes, Removing {len(X_raw['cell']) - len(X_log['cell'])} cells and {len(X_raw['gene']) - len(X_log['gene'])} genes")
    return X_log # cell * gene

def filter(X_raw, cell_cutoff=1.0, gene_cutoff=1.0, gene_filter=None, cell_filter=None):
    """Truncate barely expressed genes and cells

    Args: 
        X_raw (ndarray): [cell * gene] Raw expression matrix
        zero_ratio (float): Threshold for the truncation
    
    Returns:
        X_trunc (ndarray): [gene_trunc * cell_trunc] Truncated expression matrix
    """
    info_log.print('----------------> Truncating genes and cells ...')

    if gene_filter is None and cell_filter is None:
        gene_mask = (X_raw['expr'] > 0).mean(axis=0) >= (1-gene_cutoff) # retain the genes that are expressed in more than 1% of the cells
        cell_mask = (X_raw['expr'] > 0)[:,gene_mask].mean(axis=1) >= (1-cell_cutoff) # retain the cells where more than 1% of the genes are expressed
    else:
        gene_mask = [g in gene_filter for g in X_raw['gene']] if gene_filter is not None else range(len(X_raw['gene']))
        cell_mask = [c in cell_filter for c in X_raw['cell']] if cell_filter is not None else range(len(X_raw['cell']))
        
    X_trunc = {
            'expr': X_raw['expr'][cell_mask,:][:,gene_mask],
            'gene': X_raw['gene'][gene_mask],
            'cell': X_raw['cell'][cell_mask]
        }
    
    return X_trunc

def sort(X_trunc, num_select=2000):
    info_log.print('----------------> Sorting and selecting top genes ...')

    if num_select == -1:
        return X_trunc
    else:
        gene_idx = X_trunc['expr'].var(axis=0).argsort()[::-1][:num_select]
        return {
            'expr': X_trunc['expr'][:, gene_idx],
            'gene': X_trunc['gene'][gene_idx],
            'cell': X_trunc['cell']
        }

def log_transform(X):
    info_log.print('----------------> Log-transforming data ...')

    return {
        'expr': np.log(X['expr'] + 1),
        'expr_b4_log': X['expr'],
        'gene': X['gene'],
        'cell': X['cell']
    }
