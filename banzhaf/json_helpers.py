import json

import numpy as np


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dict_to_json(dict, filename='v_args.json'):
	"""
	Save a dictionary to a json file.
	"""
	json.dump(dict, open(filename, "w"), cls=MyEncoder, indent=4)
	print(f"Dumped JSON to {filename}")



def dump_training_results(v_args, dataset, model_type, value_type, X_train, y_train):
	v_args2 = dict()
	v_args2["scores"] = v_args["y_feature"]
	v_args2["subset_indices"] = v_args["X_feature"]
	v_args2["valuation_method"] = convert_valuation_method(value_type)
	v_args2["dataset_name"] = dataset
	v_args2["model"] = model_type
	v_args2["training_df"] =  np.column_stack([X_train, y_train])
	dict_to_json(v_args2, "../testing/output/train_results.json")



def dump_computed_semi_values(v_args, dataset, model_type, value_type, X_train, y_train, sv):
	v_args2 = dict()
	v_args2["scores"] = v_args["y_feature"]
	v_args2["subset_indices"] = v_args["X_feature"]
	v_args2["valuation_method"] = convert_valuation_method(value_type)
	v_args2["dataset_name"] = dataset
	v_args2["model"] = model_type
	v_args2["training_df"] =  np.column_stack([X_train, y_train])
	v_args2["semi_values"] = sv
	dict_to_json(v_args2, "../testing/output/computed_semi_values.json")



def convert_valuation_method(value_type):
	if value_type == "LOO":
		return "Leave One Out"
	elif value_type ==  'Shapley_Perm':
		return "Permutation Data Shapley"
	elif value_type ==  'Shapley_GT':
		return "Group Testing Shapley"
	elif value_type ==  'Banzhaf_GT':
		return "MSR Data Banzhaf"
	else:
		return "unsupported_"+value_type