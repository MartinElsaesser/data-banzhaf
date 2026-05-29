dataset_name = "cpu"

# extend path (https://stackoverflow.com/a/35273613)
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
banzhaf_dir = os.path.normpath(os.path.join(current_file_dir, "../banzhaf"))
if banzhaf_dir not in sys.path:
    print(f"add banzhaf dir to path: {banzhaf_dir}")
    sys.path.append(banzhaf_dir)

# change cwd for dataset file path resolution
print(f"switch cwd to banzhaf dir: {banzhaf_dir}")
current_cwd = os.getcwd()
os.chdir(banzhaf_dir)
from prepare_data import get_data
X, y, _, _ = get_data(dataset_name)


# export dataset as csv
import numpy as np
import pandas as pd
os.chdir(current_file_dir)  # used to dump csv into current folder
df = pd.DataFrame(np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1))
df.to_csv(f"{dataset_name}.csv")

