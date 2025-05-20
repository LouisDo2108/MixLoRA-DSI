export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI


# seq2seq_1
experiment_dir=experiments-full-lexical-ripor

data_root_dir=./mixloradsi/msmarco/d0
experiment_dir=experiments
collection_path=$data_root_dir/full_collection
queries_path=$data_root_dir/train_queries

docid_to_smtid_path=$data_root_dir/$experiment_dir/rq_smtid/docid_to_tokenids.json

# need to change for every experiment
pretrained_path=$data_root_dir/experiments/mixloradsi-pt
run_name=ripor_direct_lng_knp_seq2seq_1
output_dir=$data_root_dir/$experiment_dir/$run_name

# also need to be changed by condition
query_to_docid_path=$data_root_dir/doc2query/query_to_docid.train.json
teacher_score_path=$data_root_dir/self_neg_top100/qrel_added_qid_docids_teacher_scores.train.json

# torchrun --nproc_per_node=1 train_ripor_multistep_reranking.py \
#         --epochs=150 \
#         --run_name=$run_name \
#         --learning_rate=5e-4 \
#         --loss_type=direct_lng_knp_margin_mse \
#         --model_name_or_path=t5-base \
#         --model_type=ripor \
#         --teacher_score_path=$teacher_score_path \
#         --output_dir=$output_dir \
#         --task_names='["rank","rank_4"]' \
#         --wandb_project_name=full_lexical_ripor \
#         --use_fp16 \
#         --collection_path=$collection_path \
#         --max_length=64 \
#         --per_device_train_batch_size=128 \
#         --queries_path=$queries_path \
#         --pretrained_path=$pretrained_path \
#         --docid_to_smtid_path=$docid_to_smtid_path \
#         --query_to_docid_path=$query_to_docid_path

torchrun --nproc_per_node=1 train_ripor_multistep_reranking.py \
        --run_name=$run_name \
        --output_dir=$output_dir \
        --model_name=mixloradsi \
        --pretrained_path=$pretrained_path \
        --query_to_docid_path=$query_to_docid_path \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --teacher_score_path=$teacher_score_path \
        --collection_path=$collection_path \
        --queries_path=$queries_path \
        --save_safetensors=False \
        --save_total_limit=1 \
        --per_device_train_batch_size=512 \
        --learning_rate=5e-4 \
        --optim=adamw_8bit \
        --dataloader_num_workers=4 \
        --gradient_accumulation_steps=2 \
        --seed=42 \
        --bf16=True \
        --tf32=True \
        --num_train_epochs=150 \
        --save_steps=10000 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=1 \
        --lr_scheduler_type=linear \
        --report_to=none \
        --mixlora_config_json_path=./MixLoraDSI/config/msmarco/mixloradsi_0.01%.json \