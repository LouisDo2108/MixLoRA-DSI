#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

train_split=d0
data_root_dir=./mixloradsi/msmarco/$train_split
experiment_dir=experiments
run_name=mixloradsi-pt

model_dir=$data_root_dir/$experiment_dir/$run_name
docid_to_tokenids_path=./mixloradsi/msmarco/$train_split/experiments/rq_smtid/docid_to_tokenids.json


out_dir=$data_root_dir/$experiment_dir/$run_name/out
num_tasks=4

for (( i = 1; i <= num_tasks; i++ ))
do
        eval_split=d$i
        q_collection_paths=./mixloradsi/msmarco/$eval_split/eval_queries
        eval_qrel_path=$q_collection_paths/eval_qrel.json

        torchrun --master_port=25000 --nproc_per_node=1 eval_mixlora.py \
                --pretrained_path=$model_dir \
                --docid_to_tokenids_path=$docid_to_tokenids_path \
                --out_dir=$out_dir \
                --q_collection_paths=$q_collection_paths \
                --eval_qrel_path=$eval_qrel_path \
                --max_new_token_for_docid=8 \
                --batch_size 256 \
                --topk 10
done