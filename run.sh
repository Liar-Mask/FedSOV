#!/bin/bash
# command to run FedSOV
model=resnet
dataset=cifar100
num_bit=2048
seed=42
lr=0.01
num_users=10
CUDA_VISIBLE_DEVICES=1 python main_FedSOV.py --seed $seed --num_sign $num_users --num_bit $num_bit --num_users $num_users --dataset $dataset --model_name $model --epochs 300\
 --lr $lr --log_dir LOG_FedSOV/$num_users &

# num_users=50
# CUDA_VISIBLE_DEVICES=2 python main_FedSOV.py --seed $seed --num_sign $num_users --num_bit $num_bit --num_users $num_users --dataset $dataset --model_name $model --epochs 300\
#  --lr $lr --log_dir LOG_time/$num_users &

# num_users=100
# CUDA_VISIBLE_DEVICES=3 python main_FedSOV.py --seed $seed --num_sign $num_users --num_bit $num_bit --num_users $num_users --dataset $dataset --model_name $model --epochs 300\
#  --lr $lr --log_dir LOG_time/$num_users &

