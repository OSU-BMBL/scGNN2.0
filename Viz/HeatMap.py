import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 5})
# from util import imputation_err_heatmap

# INPUT STARTS
original_filepath = 'outputs/datasets/2.Chu/original_top_expression.csv'
imputed_filepath = 'outputs/inputs/no_bulk_2.Chu_0.1_dropout/imputed.csv'
label_filepath = 'outputs/inputs/no_bulk_2.Chu_0.1_dropout/labels.csv'
# INPUT ENDS

original = pd.read_csv(original_filepath, index_col=0, sep=None).to_numpy()
imputed = pd.read_csv(imputed_filepath, index_col=0, sep=None).to_numpy()
label = pd.read_csv(label_filepath, index_col=0, sep=None).to_numpy()

original = np.log(original + 1)
imputed = imputed.T
label = label.ravel()

print(original.shape)
print(imputed.shape)
print(label.shape)

diff = 1/(np.abs(original - imputed).T+1e-3)

ct_count = len(set(label))
diff_list = [[] for i in range(ct_count)]
for i, ct in enumerate(label):
    diff_list[ct].append(diff[i])

print(ct_count)

size_idx = np.argsort([len(np.vstack(ct)) for ct in diff_list])[::-1]
diff = np.vstack(np.array([np.vstack(ct) for ct in diff_list])[size_idx])

print(f'Median L1 Error is: {np.median(np.abs(diff))}')

plt.imshow(diff, cmap='Reds')
plt.colorbar()
plt.savefig(os.path.join('./outputs', f"imputation_error_heatmap_Chu_no_bulk.png"), dpi=1000)
plt.show()