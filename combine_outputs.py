import os
import pandas as pd

path = 'outputs'

all_filenames = []

for root, directories, files in os.walk(path, topdown=True):
    if root != path:
        _ = [all_filenames.append(os.path.join(root, f)) for f in files if f == 'all_metris.csv']

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort=False)
combined_csv = combined_csv.sort_values(['Unnamed: 0', 'Unnamed: 1']).reset_index(drop=True)

#export to csv
combined_csv.to_csv(os.path.join(path, "combined_csv.csv"), index=False, encoding='utf-8-sig')