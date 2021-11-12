"""
"""

import numpy as np

import info_log

def imputation_handler(X_imputed_sc, X_imputed_bulk, X, bigepoch, outDir):
    info_log.print('--------> Starting Imputation ...')

    avg_err_sc = np.abs(X_imputed_sc - X)
    avg_err_bulk = np.abs(X_imputed_bulk - X)
    
    avg_err_sum = avg_err_sc + avg_err_bulk + 1e-16 # for numerical stability

    integrative_prob_sc = avg_err_sc / avg_err_sum
    integrative_prob_bulk = avg_err_bulk / avg_err_sum
    # np.savetxt(f'{outDir}integrative_prob_sc_{bigepoch}.csv', integrative_prob_sc, fmt='%10.4f')
    # np.savetxt(f'{outDir}integrative_prob_bulk_{bigepoch}.csv', integrative_prob_bulk, fmt='%10.4f')

    X_imputed = integrative_prob_sc * X_imputed_sc + integrative_prob_bulk * X_imputed_bulk

    return X_imputed