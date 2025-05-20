#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

### For D0 indexing ###
data_root_dir=./mixloradsi/msmarco/d0
collection_path=$data_root_dir/full_collection/
queries_path=$data_root_dir/train_queries

experiment_dir=experiments
output_dir=$data_root_dir/$experiment_dir

eval_queries_paths='["'$data_root_dir'/eval_queries/"]'
eval_qrel_paths='["'$data_root_dir'/eval_queries/eval_qrel.json"]'


lr=5e-4
run_name=t5-self-neg-marginmse-$lr
model_dir=$data_root_dir/$experiment_dir/$run_name
pretrained_path=$model_dir/checkpoint
mmap_dir=$model_dir/mmap
out_dir=$model_dir/out

torchrun --nproc-per-node=1 index_flat.py \
    --pretrained_path=$pretrained_path \
    --index_dir=$mmap_dir \
    --mmap_dir=$mmap_dir \
    --out_dir=$out_dir \
    --encoder_type=t5seq_pretrain_encoder \
    --q_collection_paths=$eval_queries_paths \
    --eval_qrel_path=$eval_qrel_paths \
    --collection_path=$collection_path

### For adding new documents to the index ###
# data_root_dir=./mixloradsi/msmarco/d0
# collection_path=$data_root_dir/full_collection/
# queries_path=$data_root_dir/train_queries
# pretrained_path=./mixloradsi/msmarco/d0/pq/t5-self-neg-marginmse-5e-4/checkpoint

# experiment_dir=pq
# output_dir=$data_root_dir/$experiment_dir
# mmap_dir=$output_dir/mmap
# out_dir=$output_dir/out

# eval_queries_paths='["'$data_root_dir'/eval_queries/"]'
# eval_qrel_paths='["'$data_root_dir'/eval_queries/eval_qrel.json"]'

# torchrun --nproc-per-node=1 index_flat.py \
#     --pretrained_path=$pretrained_path \
#     --index_dir=$mmap_dir \
#     --mmap_dir=$mmap_dir \
#     --out_dir=$out_dir \
#     --encoder_type=t5seq_pretrain_encoder \
#     --q_collection_paths=$eval_queries_paths \
#     --eval_qrel_path=$eval_qrel_paths \
#     --collection_path=$collection_path