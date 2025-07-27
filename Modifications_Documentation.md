
# Modification documentation

This document highlights the modifications done to this project

## Custom Exports
We added functionality to export the results of the model training and the computed semi values.
Both exports are run by `banzhaf/applications.py`.

The first export saves the results of model training into `output/train_results.json`.
The second export saves the results of semi value computation into `output/computed_semi_values.json`.

Be sure to create an `output` folder in the root of the project.

The exports can than be used as imports for https://github.com/jhstaudacher/DataCGT.

## UV as a package manager
The original `requirements.txt` was a mess. There were still filesystem paths included that pointed to files on the original author's pc.
As part of cleaning up the `requirements.txt` UV was introduced as a package manager.
It is still possible to install packages via the `requirements.txt`, but we strongly discourage from doing so.

We suggest handling the installation of packages via UV.
UV makes it easy to manage dependencies and different python versions.
It automatically creates a virtual environment to prevent packages from being installed globally.

The necessary steps to setup and use UV are outlined in the following sections.

### Install UV
To install UV follow its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

### Install packages via UV
To install the right packages and python version just run:
```bash 
uv sync
```
UV will automatically setup your project and create a virtual environment.

### Running a python file with UV
To run a python file exchange `python` with `uv run`  
Instead of:
```bash
python somePythonFile.py
```
Use:
```bash
uv run somePythonFile.py
```

### Run the project with UV
As additional help to get started with UV. We provide commands for a full run:

(a full run consists of sampling subsets and computing the semi values)

#### first step: sample subsets
```bash
uv run banzhaf/sample_for_value.py --dataset pol --value_type Banzhaf_GT --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 100 --batch_size 32 --flip_ratio 0.1 --random_state 0 --lr 0.01
```

#### second step: compute and evaluate semi values
```bash
uv run banzhaf/applications.py --task mislabel_detect --dataset pol --value_type Banzhaf_GT --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 100 --batch_size 32 --flip_ratio 0.1 --random_state 0 --lr 0.01
```

## Added project documentation
We also extended the project documentation by annotating the underlying ArgumentParser.
The ArgumentParser now contains information on what input a specific option accepts.
This information is accessible through the `--help` flag.

Help is accessible on the sampling step:
```bash
uv run banzhaf/sample_for_value.py --help
```
and on the computation and evaluation step:
```bash
uv run banzhaf/applications.py --help
```

