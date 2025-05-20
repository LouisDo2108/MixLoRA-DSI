#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

current_split=1

for (( i = 0; i <= current_split; i++ ))
do
    if [ $i -eq 0 ]; then
        data_root_dir=./mixlora_dsi/msmarco/d$i
    else
        # data_root_dir=./promptdsi/msmarco/d$i
        data_root_dir=./mixlora_dsi/msmarco/d$i
    fi
    collection_path=$data_root_dir/full_collection/full_dpr
    queries_path=$data_root_dir/train_queries

    eval_queries_paths=$data_root_dir/eval_queries/
    eval_qrel_paths=$data_root_dir/eval_queries/eval_qrel.json


    pretrained_path=./mixlora_dsi/msmarco/dpr/output

    # mmap_dir=./mixlora_dsi/msmarco/d$i/experiments/dpr/mmap
    # out_dir=./mixlora_dsi/msmarco/d$i/experiments/dpr/out
    mmap_dir=./mixlora_dsi/msmarco/d$current_split/full_collection/full_dpr/mmap
    out_dir=./mixlora_dsi/msmarco/d$current_split/full_collection/full_dpr/out

    torchrun --nproc-per-node=1 index_flat_dpr.py \
        --pretrained_path=$pretrained_path \
        --index_dir=$mmap_dir \
        --mmap_dir=$mmap_dir \
        --out_dir=$out_dir \
        --encoder_type=t5seq_pretrain_encoder \
        --q_collection_paths=$eval_queries_paths \
        --eval_qrel_path=$eval_qrel_paths \
        --collection_path=$collection_path \
        --retrieve_only
done