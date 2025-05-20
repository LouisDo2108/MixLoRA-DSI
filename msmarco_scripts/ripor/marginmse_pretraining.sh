#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

data_root_dir=./mixloradsi/msmarco/d0
collection_path=$data_root_dir/full_collection/
queries_path=$data_root_dir/train_queries

experiment_dir=experiments
output_dir=./mixloradsi/msmarco/d0/$experiment_dir


eval_queries_paths=$data_root_dir/eval_queries/
eval_qrel_paths=$data_root_dir/eval_queries/eval_qrel.json

lr=5e-4

# # 1. Finetune T5 model with bm25 top 100 negative samples (M0) (Reference: PAG/full_scripts/t5_full_dense_train.sh)

pretrained_path=t5-base
run_name=t5-bm25-marginmse-$lr
teacher_score_path=$data_root_dir/bm25_top100/qrel_added_qid_docids_teacher_scores.train.json

# torchrun --nproc-per-node=1 \
#     pretrain_marginmse.py \
#         --epochs=50 \
#         --run_name=$run_name \
#         --learning_rate=$lr \
#         --loss_type=margin_mse \
#         --model_name_or_path=$pretrained_path \
#         --model_type=t5_dense \
#         --teacher_score_path=$teacher_score_path \
#         --output_dir=$output_dir \
#         --task_names='["rank"]' \
#         --collection_path=$collection_path \
#         --max_length=128 \
#         --per_device_train_batch_size=256 \
#         --queries_path=$queries_path \
#         --pretrained_path=$pretrained_path \
#         --num_decoder_layers=12 \
#         --use_fp16

# 2. Index the collection (PAG/full_scripts/t5_full_dense_evaluate_dev.sh)
# Normal dense indexing using M0 -> Results in a lot of embeddings file
model_dir=$data_root_dir/$experiment_dir/$run_name
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
mmap_dir=$model_dir/mmap
out_dir=$model_dir/out

# torchrun --nproc-per-node=1 index_flat.py \
#     --pretrained_path=$pretrained_path \
#     --index_dir=$mmap_dir \
#     --mmap_dir=$mmap_dir \
#     --out_dir=$out_dir \
#     --encoder_type=t5seq_pretrain_encoder \
#     --q_collection_paths=$eval_queries_paths \
#     --eval_qrel_path=$eval_qrel_paths \
#     --collection_path=$collection_path \
#     --index_only


# 3. Retrieve the top 100 documents for each trained query using M0 -> Results in $experiment_dir/$run_name/out/MSMARCO_TRAIN/run.json (PAG/full_scripts/t5_full_dense_evaluate_dev.sh)

model_dir=$data_root_dir/$experiment_dir/$run_name
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out

# torchrun --nproc-per-node=1 index_flat.py \
#     --pretrained_path=$pretrained_path \
#     --collection_path=$collection_path \
#     --index_dir=$mmap_dir \
#     --mmap_dir=$mmap_dir \
#     --out_dir=$out_dir \
#     --q_collection_paths=$data_root_dir/train_queries \
#     --topk=100 \
#     --encoder_type=t5seq_pretrain_encoder \
#     --retrieve_only

# # 4. Give each query-document pair in the run.json a teacher score. -> Results in $experiment_dir/$run_name/out/MSMARCO_TRAIN/qid_docids_teacher_scores.train.json (PAG/full_scripts/rerank_for_create_trainset_dev.sh)

model_dir=$data_root_dir/$experiment_dir/$run_name
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out

# torchrun --nproc-per-node=1 rerank.py \
#     --run_json_path=$out_dir/msmarco_train_d0/run.json \
#     --out_dir=$out_dir/msmarco_train_d0 \
#     --collection_path=$collection_path \
#     --q_collection_path=$queries_path \
#     --json_type=json \
#     --batch_size=2048

# # 5. Since not all top 100 docids for each query contain the relevant docids, we should add these relevant docids back if they are not in top 100 list (For NQ, for each query, obtain the top k documents with BM25 score, then find the score of the labeled documents)-> Results in $experiment_dir/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json

model_dir=$data_root_dir/$experiment_dir/$run_name
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out

# python3 t5_pretrainer/add_qrel_to_rerank_run.py \
#     --teacher_score_path=$data_root_dir/train_queries/qid_to_reldocid_to_score.json \
#     --rerank_path=$out_dir/msmarco_train_d0/qid_docids_teacher_scores.train.json \
#     --out_path=$data_root_dir/self_neg_top100/qrel_added_qid_docids_teacher_scores.train.json

# # 6. Self-negative fine-tuning M0 with the mined negatives from M0 in previous step (M1) (PAG/full_scripts/t5_full_dense_train.sh)

# # This is the self-negative score
run_name=t5-self-neg-marginmse-$lr-second-version
model_dir=$data_root_dir/$experiment_dir/t5-bm25-marginmse-$lr
pretrained_path=$model_dir/checkpoint
teacher_score_path=$data_root_dir/self_neg_top100/qrel_added_qid_docids_teacher_scores.train.json


torchrun --nproc-per-node=1 \
    pretrain_marginmse.py \
        --epochs=50 \
        --run_name=$run_name \
        --learning_rate=$lr \
        --loss_type=margin_mse \
        --model_name_or_path=$pretrained_path \
        --model_type=t5_dense \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --collection_path=$collection_path \
        --max_length=128 \
        --per_device_train_batch_size=256 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --num_decoder_layers=12 \
        --use_fp16

# # 7. Evaluate the performance of self-negative fine-tuned model
# run_name=msmarco_t5-self-neg-marginmse-$lr
# model_dir=$data_root_dir/$experiment_dir/$run_name
# pretrained_path=$model_dir/checkpoint
# index_dir=$model_dir/mmap
# mmap_dir=$model_dir/mmap
# out_dir=$model_dir/out
# eval_queries_paths=$data_root_dir/eval_queries/
# eval_qrel_paths=$data_root_dir/eval_queries/eval_qrel.json

# torchrun --nproc-per-node=1 index_flat.py \
#     --pretrained_path=$pretrained_path \
#     --collection_path=$collection_path \
#     --index_dir=$mmap_dir \
#     --mmap_dir=$mmap_dir \
#     --out_dir=$out_dir \
#     --q_collection_paths=$eval_queries_paths \
#     --eval_qrel_path=$eval_qrel_paths \
#     --topk=100 \
#     --encoder_type=t5seq_pretrain_encoder \
#     --retrieve_only
