## Fast preprocessing step:
```
cd banzhaf
python sample_for_value.py --dataset pol --value_type Banzhaf_GT --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 100 --batch_size 32 --flip_ratio 0 --random_state 0 --lr 0.01
```
NOTE:
* need to use random_state=0 and n_repeat=5 for further processing in `applications.py`



## Process semi values
```
cd banzhaf
python applications.py --task weighted_acc --dataset pol --value_type Banzhaf_GT --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 100 --batch_size 32 --flip_ratio 0 --random_state 0 --lr 0.01
```
