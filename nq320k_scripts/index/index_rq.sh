#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

split=d0
# data_root_dir=./mixlora_dsi/nq320k/$split
data_root_dir=./nq320k/$split
experiment_dir=experiments
codebook_num=8
codebook_bits=11

collection_path=$data_root_dir/full_collection/
model_dir=$data_root_dir/$experiment_dir/t5-self-neg-marginmse-5e-4
pretrained_path=$model_dir/checkpoint

mmap_dir=$data_root_dir/$experiment_dir/mmap_rq
out_dir=$data_root_dir/$experiment_dir/rq_out
q_collection_paths=$data_root_dir/eval_queries
eval_qrel_path=$data_root_dir/eval_queries/eval_qrel.json


# 2. Train the RQ codebook and index; save the index to experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_index/model.index
# torchrun --master_port=24000 --nproc_per_node=1 index_rq.py \
#     --codebook_num=$codebook_num \
#     --codebook_bits=$codebook_bits \
#     --index_dir=$mmap_dir \
#     --mmap_dir=$mmap_dir \
#     --out_dir=$out_dir \
#     --collection_path=$collection_path \
#     --pretrained_path=$pretrained_path \
#     --q_collection_paths=$q_collection_paths \
#     --eval_qrel_path=$eval_qrel_path \
#     --retrieve_only


# # 4. Save the semantic docids to a files -> Results in experiments-full-lexical-ripor/t5-full-dense-0-5e-4-12l/aq_smtid/docid_to_smtid.json
# python3 create_customized_smtid_file.py \
#     --model_dir=$model_dir \
#     --codebook_num=$codebook_num \
#     --codebook_bits=$codebook_bits \
#     --out_dir=$model_dir/rq_smtid \
#     --mmap_dir=$mmap_dir

# # 5. Extend T5's embedding layer with the semantic docids -> Results in experiments-full-lexical-ripor/t5-full-dense-0-5e-4-12l/aq_smtid/docid_to_smtid.json and experiments-full-lexical-ripor/t5-full-dense-0-5e-4-12l/extended_token_checkpoint
# python3 change_customized_embed_layer.py \
#     --model_dir=$model_dir \
#     --codebook_num=$codebook_num \
#     --codebook_bits=$codebook_bits \
#     --out_dir=$model_dir/rq_smtid \
#     --mmap_dir=$mmap_dir \
#     --extended_model_out_dir=$model_dir/extended_rq_token_checkpoint