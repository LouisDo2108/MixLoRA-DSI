export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI


task=seq2seq # Do not modify this for now
data_root_dir=./mixlora_dsi/nq320k/d0
experiment_dir=experiments
run_name=mixloradsi

model_dir=$data_root_dir/$experiment_dir/t5-self-neg-marginmse-5e-4/extended_rq_token_checkpoint
query_to_docid_path=$data_root_dir/doc2query/query_to_docid.train.json
output_dir=$data_root_dir/$experiment_dir/$run_name
docid_to_smtid_path=$data_root_dir/$experiment_dir/rq_smtid/docid_to_tokenids.json

# torchrun --nproc_per_node=1
python train_ripor_or_mixloradsi_d0.py \
        --run_name=$run_name \
        --output_dir=$output_dir \
        --model_name=mixloradsi \
        --pretrained_path=$model_dir \
        --query_to_docid_path=$query_to_docid_path \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --save_safetensors=False \
        --save_total_limit=1 \
        --per_device_train_batch_size=1024 \
        --learning_rate=1e-3 \
        --optim=adamw_8bit \
        --dataloader_num_workers=4 \
        --gradient_accumulation_steps=1 \
        --seed=42 \
        --bf16=True \
        --tf32=True \
        --max_steps=20000 \
        --save_steps=5000 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=1000 \
        --lr_scheduler_type=linear \
        --mixlora_config_json_path=./MixLoraDSI/config/mixloradsi_pretrained_d0.json \
        --report_to=none