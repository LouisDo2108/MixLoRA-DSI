#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

previous_split=d0
current_split=d1
experiment_dir=experiments

# Previous split
data_root_dir=./mixloradsi/nq320k/$previous_split
collection_path=$data_root_dir/full_collection/

flat_index_dir=./mixloradsi/nq320k/$previous_split/experiments/mmap
rq_index_dir=./mixloradsi/nq320k/$previous_split/experiments/mmap_rq

# Current split
new_data_root_dir=./mixloradsi/nq320k/$current_split
new_collection_path=$new_data_root_dir/full_collection
new_flat_index_dir=$new_data_root_dir/$experiment_dir/mmap
new_rq_index_dir=$new_data_root_dir/$experiment_dir/mmap_rq
out_dir=$new_data_root_dir/$experiment_dir/rq_out
q_collection_paths=$new_data_root_dir/eval_queries
eval_qrel_path=$new_data_root_dir/eval_queries/eval_qrel.json

# 1. Add to the rq index
# torchrun --master_port=26000 --nproc_per_node=1 add_rq.py \
#     --codebook_num=8 \
#     --codebook_bits=11 \
#     --collection_path=$collection_path \
#     --pretrained_path=./mixloradsi/nq320k/d0/experiments/t5-self-neg-marginmse-5e-4/checkpoint \
#     --flat_index_dir=$flat_index_dir \
#     --rq_index_dir=$rq_index_dir \
#     --new_collection_path=$new_collection_path \
#     --new_flat_index_dir=$new_flat_index_dir \
#     --new_rq_index_dir=$new_rq_index_dir \
#     --q_collection_paths=$q_collection_paths \
#     --eval_qrel_path=$eval_qrel_path \
#     --out_dir=$out_dir


# 2. Create a file that maps docids to RQ code
python3 create_customized_smtid_file_cl.py \
    --codebook_num=8 \
    --codebook_bits=11 \
    --new_flat_index_dir=$new_flat_index_dir \
    --new_rq_index_dir=$new_rq_index_dir \
    --previous_out_dir=$data_root_dir/$experiment_dir/rq_smtid \
    --new_out_dir=$new_data_root_dir/$experiment_dir/rq_smtid