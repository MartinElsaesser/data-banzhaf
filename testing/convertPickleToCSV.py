# import a pickle file and convert it to a csv file
import pandas as pd
import pickle
import os
import argparse
import numpy as np
import sys


def convert_pickle_to_csv(pickle_filepath, csv_file):
	# Load the pickle file
	with open(pickle_filepath, 'rb') as f:
		data = pickle.load(f, )

	# print(np.array(data["X_feature"]).shape)
	# Convert the data to a DataFrame
	df_X = pd.DataFrame(data["X_feature"])

	# Save the DataFrame to a CSV file
	df_X.to_csv(csv_file, index=False)

convert_pickle_to_csv("./banzhaf/result/Banzhaf_GT_pol_MLP_Ndata200_Nval200_Nsample100_BS32_LR0.01_Nrepeat5_FR0.0_Seed0.data", "./testing/debug.csv")