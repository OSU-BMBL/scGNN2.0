"""
"""

import os

import numpy as np
import scipy.sparse as sp
import pandas as pd

import info_log


def dropout(X_sc, args):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero, X_zero_b4_log = X_sc['expr'].copy(), X_sc['expr_b4_log'].copy()

    if not args.dropout_prob:
        return {'expr': X_zero, 'expr_b4_log': X_zero_b4_log}, None

    info_log.print('--------> Applying dropout for imputation testing ...')
    
    rate = args.dropout_prob
    seed = args.seed

    # If the input is a dense matrix
    if isinstance(X_zero, np.ndarray):
        # X_zero = np.copy(X['expr'])
        # select non-zero subset
        i, j = np.nonzero(X_zero)
    # If the input is a sparse matrix
    else:
        # X_zero = scipy.sparse.lil_matrix.copy(X)
        # select non-zero subset
        i, j = X_zero.nonzero()

    np.random.seed(seed)
    # changes here:
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(rate * len(i))), replace=False)
    # X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
    X_zero[i[ix], j[ix]] = 0.0
    X_zero_b4_log[i[ix], j[ix]] = 0.0

    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    
    return {'expr': X_zero, 'expr_b4_log': X_zero_b4_log}, (i, j, ix) # new matrix with dropout same shape as X, row index of non zero entries, column index of non zero entries, index for entries in list i and j that are set to zero

# IMPUTATION METRICS
# Revised freom Original version in scVI
# Ref:
# https://github.com/romain-lopez/scVI-reproducibility/blob/master/demo_code/benchmarking.py

def imputation_error_handler(X_imputed, X_orig, dropout_prob, dropout_info=None):

    error_median, error_median_inv = None, None
    if dropout_prob:
        error_mean, error_median, error_min, error_max = imputation_error_dropout(X_imputed, X_orig, *dropout_info)
        error_mean_inv, error_median_inv, error_min_inv, error_max_inv = imputation_error_inverse(X_imputed, X_orig, *dropout_info)
    error_mean_entire, error_median_entire, error_min_entire, error_max_entire = imputation_error_entire(X_imputed, X_orig)
    
    return error_median, error_median_inv, error_median_entire

def imputation_error_dropout(X_mean, X, i, j, ix):
    """
    X_mean: imputed dataset [gene * cell]
    X: original dataset [gene * cell]
    X_zero: zeros dataset, does not need 
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    # info_log.print('--------> Computing imputation error ...')

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
        result = np.abs(x - y)
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = sp.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        result = np.abs(x - yuse)
        
    # return np.median(np.abs(x - yuse))
    return np.mean(result), np.median(result), np.min(result), np.max(result)

def imputation_error_inverse(X_mean, X, i, j, ix):
    """
    X_mean: imputed dataset [gene * cell]
    X: original dataset [gene * cell]
    X_zero: zeros dataset, does not need 
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets only at indices that are NOT given in ix
    """
    # info_log.print('--------> Computing imputation error on none dropout data ...')

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[-ix], j[-ix]
        x, y = X_mean[all_index], X[all_index]
        result = np.abs(x - y)
    # If the input is a sparse matrix
    else:
        all_index = i[-ix], j[-ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = sp.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        result = np.abs(x - yuse)
        
    # return np.median(np.abs(x - yuse))
    return np.mean(result), np.median(result), np.min(result), np.max(result)

def imputation_error_entire(X_mean, X):
    """
    X_mean: imputed dataset [gene * cell]
    X: original dataset [gene * cell]
    ========
    returns:
    median L1 distance between datasets
    """
    # info_log.print('--------> Computing imputation error on the entire matrix ...')

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        result = np.abs(X_mean - X)
    # If the input is a sparse matrix
    else:
        yuse = sp.lil_matrix.todense(X)
        yuse = np.asarray(yuse).reshape(-1)
        result = np.abs(X_mean - yuse)
        
    # return np.median(np.abs(x - yuse))
    return np.mean(result), np.median(result), np.min(result), np.max(result)