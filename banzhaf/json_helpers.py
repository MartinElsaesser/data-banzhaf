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


def dict_to_json(dict, filename="v_args.json"):
    """
    Save a dictionary to a json file.
    """
    json.dump(dict, open(filename, "w"), cls=MyEncoder, indent=4)
    print(f"Dumped JSON to {filename}")


def dump_computed_semi_values(v_args, dataset, model_type, value_type, sv, i):
    v_args2 = {
        "scores": v_args["y_feature"],
        "subset_indices": v_args["X_feature"],
        "valuation_method": rename_valuation_method_for_R(value_type),
        "dataset_name": dataset,
        "model": model_type,
        "semi_values": sv,
    }
    dict_to_json(v_args2, f"../output/computed_semi_values_{i}.json")


def rename_valuation_method_for_R(value_type):
    if value_type == "LOO":
        return "Leave One Out"
    elif value_type == "Shapley_Perm":
        return "Permutation Data Shapley"
    elif value_type == "Shapley_GT":
        return "Group Testing Shapley"
    elif value_type == "Banzhaf_GT":
        return "MSR Data Banzhaf"
    else:
        return "unsupported_" + value_type
