import numpy as np
import pandas as pd

import auto_encoders.graph_AE as gae
import util

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import os

import umap

np.random.seed(2020)
plt.rcParams['figure.dpi'] = 150

data = '2.Chu'

path = f"outputs/inputs/bulk_2.Chu_0.1_dropout/"
true_label_path = f'outputs/datasets/{data}/top_cell_labels.csv'

edgelist_path = path+"graph_edgeList.csv"
edge_list = np.genfromtxt(edgelist_path, delimiter=',',skip_header=1)[:,1:]
edge_list = [[int(start), int(end), weight] for start, end, weight in edge_list]
print(edge_list[:5])

label_path = path+"labels.csv"
c_label = np.genfromtxt(label_path, delimiter=',',skip_header=1, dtype=int)[:,1:].reshape(-1)
print(c_label.shape)

true_label = np.genfromtxt(true_label_path, delimiter=',',skip_header=1, dtype=int)[:,1:].reshape(-1)
print(true_label.shape)

embed_path = path+"graph_embedding.csv"
z = np.genfromtxt(embed_path, delimiter=',', skip_header=1)[:,1:]
print(z.shape)


G = nx.DiGraph()
G.add_weighted_edges_from(edge_list)

# attr = {i:f for i,f in enumerate(feature)}
# nx.set_node_attributes(G, attr, 'feature')
edgewidth = [G.get_edge_data(u, v)['weight'] for u, v in G.edges()]
pos = nx.get_node_attributes(G,"feature")
# nodesize = [np.linalg.norm(pos[n])*200 for n in G.nodes()]
nodecolor = [c_label[n] for n in G.nodes()]

reducer = umap.UMAP(random_state=2021)
embedding = reducer.fit_transform(z)
pos = {n: embedding[n] for n in G.nodes()}

sns.kdeplot(edgewidth)
edgewidth_log = np.log(edgewidth)
edgewidth_log = edgewidth_log + np.abs(min(edgewidth_log))
sns.kdeplot(edgewidth_log)

edgescale = 0.03

nx.draw_networkx(G, pos=pos, width=edgewidth_log*edgescale, with_labels=False, arrows=0, node_size=5, node_color=nodecolor, connectionstyle="arc3,rad=0.1")
plt.axis('equal')
plt.savefig(os.path.join(path, f"cell_graph.png"), dpi=1000)
plt.show()