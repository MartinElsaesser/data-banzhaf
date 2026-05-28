# add root dir as import path: https://stackoverflow.com/a/35273613
import os
import sys

module_path = "/workspaces/data-banzhaf-python/banzhaf"
print(f"using module path: {module_path}")
os.chdir(module_path)   # needed for correct dataset file path resolution
if module_path not in sys.path:
    sys.path.append(module_path)

from prepare_data import get_data
import numpy as np
import pandas as pd

X, y, _, _ = get_data("cpu")

df = pd.DataFrame(np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1))

df.to_csv("cpu.csv")
print(0)
