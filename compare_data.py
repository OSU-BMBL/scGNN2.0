import numpy as np
import pandas as pd

file_a = "./outputs/pickle.csv"
file_b = "./outputs/T2000.csv"

table_a = pd.read_csv(file_a, index_col=0, sep=None).sort_index()
table_b = pd.read_csv(file_b, index_col=0, sep=None).sort_index()

print(table_a.head)
print(table_b.head)

idx_a = table_a.index.to_numpy()
idx_b = table_b.index.to_numpy()

gene_diff = [g for g in idx_a if g not in idx_b]

print(len(gene_diff))


a = table_a.to_numpy()[:,:-1]
b = table_b.to_numpy()[:,1:]

diff = np.sum(np.abs(a-b))
print(diff)
