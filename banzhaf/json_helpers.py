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


def dump_training_results(
    df_train, df_val, X_subset_indices, model_scores_for_subset_indices, value_type, dataset, model_type, sv_baseline, random_state
):
    out = dict()
    out["df_train"] = df_train
    out["df_val"] = df_val
    out["X_subset_indices"] = X_subset_indices
    out["model_scores_for_subset_indices,"] = (model_scores_for_subset_indices,)
    out["value_type"] = value_type
    out["value_type_R"] = convert_valuation_method(value_type)
    out["dataset"] = dataset
    out["model_type,"] = (model_type,)
    out["sv_baseline"] = sv_baseline
    out["random_state"] = random_state
    dict_to_json(out, "../output/train_results.json")


def dump_computed_semi_values(v_args, dataset, model_type, value_type, X_train, y_train, sv):
    v_args2 = dict()
    v_args2["scores"] = v_args["y_feature"]
    v_args2["subset_indices"] = v_args["X_feature"]
    v_args2["valuation_method"] = convert_valuation_method(value_type)
    v_args2["dataset_name"] = dataset
    v_args2["model"] = model_type
    v_args2["training_df"] = np.column_stack([X_train, y_train])
    v_args2["semi_values"] = sv
    dict_to_json(v_args2, "../output/computed_semi_values.json")


def convert_valuation_method(value_type):
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
