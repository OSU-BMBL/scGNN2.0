"""

"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

import os
import pickle as pkl

import info_log

def sc_handler(args):
    dir_path = os.path.join(args.load_dataset_dir, args.load_dataset_name)
    
    if args.load_use_benchmark:
        # return load_benchmark(dir_path, dataset_name=args.load_dataset_name)
        return load_dense(
            os.path.join(dir_path, f'original_top_expression.csv'),
            is_cell_by_gene = False
        )
    elif args.load_seurat_object:
        info_log.print('--------> Loading from seurat object ...')
        return load_dense(
            args.load_seurat_object,
            is_cell_by_gene = True
        )
    elif args.load_rdata:
        info_log.print('--------> Loading from rdata ...')
        return load_rdata(
            args.load_rdata,
            is_cell_by_gene = True
        )
    elif args.load_from_10X:
        info_log.print('--------> Loading from 10X data ...')
        return load_from_10X(
            args.load_from_10X,
            is_cell_by_gene = True
        )
    else:
        info_log.print('--------> Loading sc raw expression ...')
        return load_dense(
            os.path.join(dir_path, args.load_sc_dataset),
            is_cell_by_gene = False
        )

def load_rdata(rdata_path,is_cell_by_gene = True,dtype=float,has_cell_name=True,has_gene_name=True):
    import pyreadr
    rdata = pyreadr.read_r(rdata_path)
    df=rdata['df']
    expr = df.to_numpy().astype(dtype)

    # Get cell and gene names if present, otherwise use an array of 0's as placeholder
    rows = np.zeros(df.shape[0])
    columns = np.zeros(df.shape[1])
    cell = rows if is_cell_by_gene else columns
    gene = columns if is_cell_by_gene else rows
    if has_cell_name:
        cell = df.index.to_numpy() if is_cell_by_gene else df.columns.to_numpy()
    if has_gene_name:
        gene = df.columns.to_numpy() if is_cell_by_gene else df.index.to_numpy()

    X_dense = {
        'expr': expr if is_cell_by_gene else expr.T,
        'gene': gene,
        'cell': cell
    }

    info_log.print(f"----------------> Matrix has {len(X_dense['cell'])} cells and {len(X_dense['gene'])} genes")
    return X_dense


def load_from_10X(dir_path,is_cell_by_gene=True, has_gene_name=True, has_cell_name=True, dtype=float, kwargs=None):
    info_log.print('----------------> Reading matrix (dense) ...')

    kwargs = {'index_col': 0, 'sep': None} if kwargs is None else kwargs
    import scanpy as sc
    adata = sc.read_10x_mtx(
        dir_path,  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)  
    expr = adata.X.todense().astype(dtype)

    # Get cell and gene names if present, otherwise use an array of 0's as placeholder
    rows = np.zeros(expr.shape[0])
    columns = np.zeros(expr.shape[1])
    cell = rows if is_cell_by_gene else columns
    gene = columns if is_cell_by_gene else rows
    if has_cell_name:
        cell = np.array(adata.obs.index) if is_cell_by_gene else np.array(adata.var.index)
    if has_gene_name:
        gene = np.array(adata.var.index) if is_cell_by_gene else np.array(adata.obs.index)

    X_dense = {
        'expr': expr if is_cell_by_gene else expr.T,
        'gene': gene,
        'cell': cell
    }

    info_log.print(f"----------------> Matrix has {len(X_dense['cell'])} cells and {len(X_dense['gene'])} genes")
    return X_dense

def bulk_handler(args):
    info_log.print('--------> Loading bulk data ...')

    dir_path = os.path.join(args.load_dataset_dir, args.load_dataset_name)

    if args.load_use_benchmark:
        return load_dense(
            os.path.join(dir_path, f'{args.load_dataset_name}_bulk.csv'),
            is_cell_by_gene = False
        )
    else:
        return load_dense(
            os.path.join(dir_path, args.load_bulk_dataset),
            is_cell_by_gene = False
        )

def LTMG_handler(X_sc,args):
    '''
    Read LTMG matrix as the regularizor. nonsparseMode
    '''
    info_log.print('--------> Loading LTMG matrix ...')

    dir_path = os.path.join(args.load_dataset_dir, args.load_dataset_name)
    
    if args.run_LTMG:
        return load_dense(
            os.path.join(args.output_dir, 'preprocessed_data', f'LTMG_{args.dropout_prob}.csv'),
            is_cell_by_gene = False
        )['expr']
    elif args.load_use_benchmark:
        return load_dense(
            os.path.join(dir_path, f'LTMG_{args.dropout_prob}.csv'),
            is_cell_by_gene = False
        )['expr']
    elif args.load_LTMG is None:
        return np.zero_like(X_sc['expr'])
    else:
        return load_dense(
            os.path.join(dir_path, args.load_LTMG),
            is_cell_by_gene = False
        )['expr']

def load_benchmark(dir_path, dataset_name):
    # load the data: x, tx, allx, graph
    info_log.print('--------> Loading benchmarking sc data ...')
    
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, f'ind.{dataset_name}')
    dense_expr_path = os.path.join(dir_path, 'T2000_expression.csv') # currently get gene names from the expr matrix

    # Read expression file
    suffix = ['x', 'tx', 'allx']
    objects = []
    for i in range(len(suffix)):
        with open(f'{dataset_path}.{suffix[i]}', 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    
    # Read in gene names
    dense_expr = load_dense(dense_expr_path, is_cell_by_gene=False)
    gene = dense_expr['gene']
    cell = dense_expr['cell']

    # Format
    _, tx, allx = tuple(objects)
    features = sp.vstack((tx, allx)).toarray() # features is cell * gene
    # features = sp.vstack((allx, tx)).toarray() # features is cell * gene, this is incorrect
    
    # Check if the pickled expression file is the same as T2000_expression.txt/csv file -> doesn't seem to be the same
    # check_diff = np.sum(np.sum(np.abs(features - dense_expr['expr'])))
    # info_log.print(f'\n> check_diff = {check_diff}')
    # exit()
    
    X_raw = {
        'expr': features.astype(float),
        'cell': cell,
        'gene': gene
        }
    
    info_log.print(f"--------> Benchmark sc data has {X_raw['expr'].shape[0]} cells, {X_raw['expr'].shape[1]} genes")
    return X_raw # cell * gene

def load_dense(file_path, is_cell_by_gene=True, has_gene_name=True, has_cell_name=True, dtype=float, kwargs=None):
    """Load the data from file

    Args: 
        args: The file name of the raw expression data, including its file path
    
    Returns:
        X_dense (dict): [cell_raw * gene_raw] Raw expression matrix
        gene (list): [gene_raw] List of gene names
        cell (list): [cell_raw] List of cell names
    """
    info_log.print('----------------> Reading matrix (dense) ...')

    kwargs = {'index_col': 0, 'sep': None} if kwargs is None else kwargs

    X = pd.read_csv(file_path, **kwargs)
    expr = X.to_numpy().astype(dtype)
    
    # Get cell and gene names if present, otherwise use an array of 0's as placeholder
    rows = np.zeros(X.shape[0])
    columns = np.zeros(X.shape[1])
    cell = rows if is_cell_by_gene else columns
    gene = columns if is_cell_by_gene else rows
    if has_cell_name:
        cell = X.index.to_numpy() if is_cell_by_gene else X.columns.to_numpy()
    if has_gene_name:
        gene = X.columns.to_numpy() if is_cell_by_gene else X.index.to_numpy()
    
    X_dense = {
        'expr': expr if is_cell_by_gene else expr.T,
        'gene': gene,
        'cell': cell
    }

    info_log.print(f"----------------> Matrix has {len(X_dense['cell'])} cells and {len(X_dense['gene'])} genes")
    return X_dense

def cell_type_labels(args, cell_filter):
    info_log.print('--------> Loading true cell type labels ...')
    dir_path = os.path.join(args.load_dataset_dir, args.load_dataset_name)

    if args.load_use_benchmark:
        ct_labels = pd.read_csv(
            os.path.join(dir_path, f'top_cell_labels.csv'), index_col=0
        ).iloc[:, 0].to_numpy()
    else:
        y = pd.read_csv(
            os.path.join(dir_path, args.load_cell_type_labels), index_col=0, sep=None)
        cells = np.array(y.index.tolist())
        # info_log.print(f"cells has shape: {cells.shape}")
        # info_log.print(cells)
        ct_labels = y.iloc[:, 0].to_numpy().ravel()
        # info_log.print(f"ct_labels has shape: {ct_labels.shape}")
        # info_log.print(ct_labels)
        ct_labels = ct_labels[[c in cell_filter for c in cells]]

    info_log.print(f"--------> {len(ct_labels)} ground-truth cell type labels loaded")
    return np.unique(ct_labels, return_inverse=True)[1] # this is to convert potential non-numeric labels to an int array

def load_CCC():
    pass