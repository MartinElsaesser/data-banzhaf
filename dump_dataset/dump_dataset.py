OpenML_dataset = ['fraud', 'apsfail', 'click', 'phoneme', 'wind', 'pol', 'creditcard', 'cpu', 'vehicle', '2dplanes']

# extend path (https://stackoverflow.com/a/35273613)
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
banzhaf_dir = os.path.normpath(os.path.join(current_file_dir, "../banzhaf"))
if banzhaf_dir not in sys.path:
    print(f"add banzhaf dir to path: {banzhaf_dir}")
    sys.path.append(banzhaf_dir)

def export_dataset(dataset_name):
    current_cwd = os.getcwd()
    os.chdir(banzhaf_dir) # change cwd for dataset file path resolution
    from prepare_data import get_data
    X, y, _, _ = get_data(dataset_name)

    # export dataset as csv
    import numpy as np
    import pandas as pd
    os.chdir(current_file_dir)  # change cwd for dumping csv into current folder
    df = pd.DataFrame(np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1))
    export_dir="exports"
    os.makedirs(export_dir, exist_ok=True)
    df.to_csv(f"{export_dir}/{dataset_name}.csv")
    os.chdir(current_cwd) # restore cwd

for dataset_name in OpenML_dataset:
    export_dataset(dataset_name)

