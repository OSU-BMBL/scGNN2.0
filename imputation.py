"""
"""

import numpy as np

import info_log

def imputation_handler(X_imputed_sc, X_imputed_bulk, X, args, param):
    info_log.print('--------> Starting Imputation ...')

    outDir = args.output_dir
    epoch_num = param['epoch_num']
    
    X_imputed_bulk[X != 0] = X[X != 0]

    avg_err_sc = X_imputed_sc - X # np.abs(X_imputed_sc - X)
    avg_err_bulk = X_imputed_bulk - X # np.abs(X_imputed_bulk - X)
    
    avg_err_sum = avg_err_sc + avg_err_bulk + 1e-16 # for numerical stability

    integrative_prob_sc = 1 - avg_err_sc / avg_err_sum
    integrative_prob_bulk = 1 - avg_err_bulk / avg_err_sum
    
    # np.savetxt(f'{outDir}integrative_prob_sc_{epoch_num}.csv', integrative_prob_sc, fmt='%10.4f')
    # np.savetxt(f'{outDir}integrative_prob_bulk_{epoch_num}.csv', integrative_prob_bulk, fmt='%10.4f')

    X_imputed = integrative_prob_sc * X_imputed_sc + integrative_prob_bulk * X_imputed_bulk


    param['X_imputed_sc'] = X_imputed_sc
    param['X_imputed_bulk'] = X_imputed_bulk

    return X_imputed