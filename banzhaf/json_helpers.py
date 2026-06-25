import json

import numpy as np
import pandas as pd


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


def dict_to_json(dict, file_path):
    """
    Save a dictionary to a json file.
    """
    json.dump(dict, open(file_path, "w"), cls=MyEncoder, indent=4)
    print(f"Dumped JSON to {file_path}")


def load_json_as_dict(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def dump_computed_semi_values(v_args, dataset, model_type, value_type, sv, i):
    if i != 0:
        # expect this function to be called 5 times, only export on first call (i==0)
        return
    train_results_path = "../output/train_results.json"
    train_results = load_json_as_dict(train_results_path)
    train_results["semi_values"] = ["{:0.20f}".format(x) for x in sv]
    dict_to_json(train_results, train_results_path)


def rename_valuation_method_for_R(value_type):
    if value_type == "LOO":
        return "Leave One Out"
    elif value_type == "Shapley_Perm":
        return "Permutation Shapley"
    elif value_type == "Banzhaf_GT":
        return "MSR Data Banzhaf"
    else:
        return "unsupported_" + value_type
