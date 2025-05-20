#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/ft49/thuy0050/miniconda/conda/envs/mixloradsi/lib
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixloradsi
cd ./MixLoraDSI

i=4
current_split=d$i

data_root_dir=./mixloradsi/msmarco/$current_split
pretrained_path=./mixloradsi/msmarco/d0/experiments/t5-self-neg-marginmse-5e-4/checkpoint
collection_path=$data_root_dir/full_collection
output_dir=$data_root_dir/experiments/bm25
mmap_dir=$output_dir/mmap
out_dir=$output_dir/out

for ((eval_i=0; eval_i<=$i; eval_i++))
    do
        eval_split=d$eval_i
        eval_queries_paths=./mixloradsi/msmarco/$eval_split/eval_queries
        eval_qrel_paths=$eval_queries_paths/eval_qrel.json

        torchrun --nproc-per-node=1 index_flat.py \
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