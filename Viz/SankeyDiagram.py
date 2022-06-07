import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import numpy as np
import pandas as pd

from collections import Counter

# INPUT STARTS
file_list = [
    'outputs/outputs_v1.2.f.2.6/10241899_2.Chu_0.1_dropout/labels.csv',
    'outputs/datasets/2.Chu/top_cell_labels.csv',
    'outputs/inputs/bulk_2.Chu_0.1_dropout/labels.csv',
]
# INPUT ENDS

node_labels = []
node_links = []

for i, label_file in enumerate(file_list):
    c_label = pd.read_csv(label_file)
    labels = c_label.iloc[:,1]
    
    source_tmp = chr(65+i) + labels.astype(str)
    
    labels_tmp = pd.unique(source_tmp)
    labels_tmp.sort()
    labels_tmp = labels_tmp.tolist()

    node_labels.append(labels_tmp)
    node_links.append(source_tmp.tolist())


source = node_links[:-1]
source = [item for sublist in source for item in sublist]

target = node_links[1:]
target = [item for sublist in target for item in sublist]

node_labels = [item for sublist in node_labels for item in sublist]
node_dict = {y:x for x, y in enumerate(node_labels)}

links = [(s,t) for s, t in zip(source, target)]

link_count_dict = Counter(links)
links = list(link_count_dict.keys())
value = [link_count_dict[l] for l in links]
source = [s for s, t in links]
target = [t for s, t in links]

source_node = [node_dict[x] for x in source]
target_node = [node_dict[x] for x in target]


fig = go.Figure( 
    data=[go.Sankey( # The plot we are interest
        # This part is for the node information
        node = dict( 
            label = node_labels
        ),
        # This part is for the link information
        link = dict(
            source = source_node,
            target = target_node,
            value = value
        ))])

plot(fig,
     image_filename='sankey_plot_1', 
     image='png', 
     image_width=2100, 
     image_height=1400
)

# And shows the plot
fig.show()