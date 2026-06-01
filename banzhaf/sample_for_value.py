import argparse
import copy
import datetime as dt
import pickle
import random

import config

# general
import numpy as np
import numpy.typing as npt
import torch
from helper import *
from json_helpers import dict_to_json, rename_valuation_method_for_R
from prepare_data import *
from utility_func import *

big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

parser = argparse.ArgumentParser("")


parser.add_argument("--dataset", type=str, choices=big_dataset + OpenML_dataset + ["covertype"])
parser.add_argument("--value_type", type=str, choices=["Banzhaf_MC", "Shapley_Perm", "Banzhaf_GT", "Shapley_GT", "LOO", "KNN"])
parser.add_argument("--model_type", type=str, choices=["MLP", "ResNet18", "ResNet50", "DenseNet", "SmallCNN", "LargeCNN", "Logistic"])
parser.add_argument("--n_data", type=int, default=500, help="rows in x_train/y_train")
parser.add_argument("--n_val", type=int, default=2000, help="rows in x_val/y_val")
parser.add_argument("--n_repeat", type=int, default=1, help="repeat model-training n times per utility function evaluation")
parser.add_argument("--n_sample", type=int, help="intent not clear; should be a mulitple of n_data")
parser.add_argument("--random_state", type=int, default=1, help="seed to init random number generators")
parser.add_argument("--flip_ratio", type=float, default=0, help="percentage of values to intentionally falsify in y_train")
parser.add_argument("--batch_size", type=int, default=8, help="batch size for model-training")
parser.add_argument("--lr", type=float, default=1e-3, help="model learning rate")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

dataset: str = args.dataset
value_type: str = args.value_type
model_type: str = args.model_type
n_data: int = args.n_data
n_val: int = args.n_val
n_repeat: int = args.n_repeat
n_sample: int = args.n_sample
random_state: int = args.random_state
flip_ratio: float = args.flip_ratio
batch_size: int = args.batch_size
lr: float = args.lr


verbose = 0
if args.debug:
    verbose = 1


save_dir = "result/"


if dataset in big_dataset + OpenML_dataset:
    save_name = save_dir + "{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_Seed{}.data".format(
        value_type, dataset, model_type, n_data, n_val, n_sample, batch_size, lr, n_repeat, flip_ratio, random_state
    )
else:
    save_name = save_dir + "{}_{}_{}_Ndata{}_Nval{}_Nsample{}_FR{}.data".format(value_type, dataset, model_type, n_data, n_val, n_sample, flip_ratio)


u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose)
utility_func_mult = lambda a, b, c, d: sample_utility_multiple(a, b, c, d, u_func, n_repeat)

x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)

utility_func_args = (x_train, y_train, x_val, y_val)

n_class = len(np.unique(y_val))
sv_baseline = 1.0 / n_class

if random_state != -1:
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def process_yfeature(y_feature):
    y_feature = np.array(y_feature)
    if n_repeat == 1:
        y_feature = y_feature.reshape(-1)
    return y_feature


df_train = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
df_val = np.concatenate((x_val, y_val[:, np.newaxis]), axis=1)


def export_training_results(args: dict):
    now = (dt.datetime.now()).isoformat(sep=" ", timespec="seconds")
    out = {
        "dataset_name": dataset,
        "model": model_type,
        "valuation_method": rename_valuation_method_for_R(value_type),
        "valuation_method_python": value_type,
        "timestamp": now,
        "training_df": df_train,
        "validation_df": df_val,
        "sv_baseline": sv_baseline,
        "random_state": random_state,
    }
    dict_to_json(out | args, "../output/train_results.json")


def export_results_basic(scores: npt.NDArray, subset_indices: list):
    if scores.ndim == 1:
        print("scores are 1d, all good")
    elif scores.ndim == 2:
        scores = scores[:, 0]
        print("scores are 2d, aggregate nested arrays by mean")
    else:
        raise ValueError(f"expected scores to have 1 or 2 dimensions, instead found: {'scores'.ndim} dimensions")
    export_training_results({"scores": scores, "subset_indices": subset_indices})


def export_results_loo(scores: npt.NDArray, subset_indices: list, u_total: npt.NDArray):
    # add x_train indices to subset_indices
    subset_indices = copy.deepcopy(subset_indices)
    subset_indices.append(np.arange(n_data))

    # add u_total to scores
    if scores.ndim == 1:
        scores = np.concatenate([scores, np.array([u_total])], axis=0)
        print("scores are 1d, all good - only add u_total to scores")
    elif scores.ndim == 2:
        scores = np.concatenate([scores, u_total[np.newaxis, :]], axis=0)
        scores = scores[:, 0]
        print("scores are 2d, aggregate nested arrays by mean and add u_total to scores")
    else:
        raise ValueError(f"expected scores to have 1 or 2 dimensions, instead found: {'scores'.ndim} dimensions")
    export_training_results({"scores": scores, "subset_indices": subset_indices})


if value_type == "Banzhaf_MC":
    n_sample_per_data = int(n_sample / n_data)
    save_arg = {}
    for target_ind in range(n_data):
        utility_set_tgt = sample_utility_banzhaf_mc(n_sample_per_data, utility_func_mult, utility_func_args, target_ind)
        save_arg[target_ind] = utility_set_tgt

elif value_type == "Shapley_Perm":
    n_perm = int(n_sample / n_data)
    X_feature_test, y_feature_test = sample_utility_shapley_perm(n_perm, utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {"X_feature": X_feature_test, "y_feature": y_feature_test}
    export_results_basic(scores=y_feature_test, subset_indices=X_feature_test)

elif value_type == "Banzhaf_GT":
    X_feature_test, y_feature_test = sample_utility_banzhaf_gt(
        n_sample, utility_func_mult, utility_func_args, dummy=False
    )  # dummy=False is needed for python comparison, originally it was set to True
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {"X_feature": X_feature_test, "y_feature": y_feature_test}
    export_results_basic(scores=y_feature_test, subset_indices=X_feature_test)

elif value_type == "Shapley_GT":
    X_feature_test, y_feature_test = sample_utility_shapley_gt(n_sample, utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {"X_feature": X_feature_test, "y_feature": y_feature_test}
    export_results_basic(scores=y_feature_test, subset_indices=X_feature_test)

elif value_type == "LOO":
    X_feature_test, y_feature_test, u_total = sample_utility_loo(utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    u_total = np.array(u_total)
    if n_repeat == 1:
        u_total = u_total[0]
    save_arg = {"X_feature": X_feature_test, "y_feature": y_feature_test, "u_total": u_total}
    export_results_loo(scores=y_feature_test, subset_indices=X_feature_test, u_total=u_total)

elif value_type == "KNN":
    sv = knn_shapley(x_train, y_train, x_val, y_val, K=10)
    export_training_results({"knn_shapley": sv})

save_arg["sv_baseline"] = sv_baseline
save_arg["n_data"] = n_data


pickle.dump(save_arg, open(save_name, "wb"))
