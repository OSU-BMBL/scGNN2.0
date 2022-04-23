import numpy as np
import pandas as pd

data = '1'

# file_a = f"./outputs/{data}/T2000_expression.csv"
file_a = f"./outputs/{data}/dropout_top_expression.csv"
file_b = f"./outputs/{data}/original_top_expression.csv"

# file_a = f"./outputs/{data}/T2000_LTMG.csv"
# file_b = f"./outputs/{data}/LTMG_0.0.csv"

table_a = pd.read_csv(file_a, index_col=0, sep=None)#.sort_index()
table_b = pd.read_csv(file_b, index_col=0, sep=None)#.sort_index()

print(table_a.head(10))
print(table_b.head(10))

print(table_a.var(axis=1))
print(table_b.var(axis=1))

idx_a = table_a.index.to_numpy()
idx_b = table_b.index.to_numpy()

gene_diff = [g for g in idx_a if g not in idx_b]

print(len(gene_diff))

a = table_a.to_numpy()
b = table_b.to_numpy()

# print(a[:10,:10])
# print(b[:10,:10])

diff = np.sum(np.abs(a-b))
print(diff)

print(a.sum()-b.sum())
