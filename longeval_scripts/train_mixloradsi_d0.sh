source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

export CUDA_VISIBLE_DEVICES="1"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TQDM_DISABLE=1


task=seq2seq # Do not modify this for now
data_root_dir=./mixloradsi/longeval/2022-10
experiment_dir=experiments
run_name=mixloradsi

model_dir=$data_root_dir/$experiment_dir/t5-bm25-marginmse-5e-4/extended_rq_token_checkpoint
query_to_docid_path=$data_root_dir/doc2query/query_to_docid.train.json
output_dir=$data_root_dir/$experiment_dir/$run_name
docid_to_smtid_path=$data_root_dir/$experiment_dir/rq_smtid/docid_to_tokenids.json

python train_ripor_or_mixloradsi_d0_longeval.py \
        --run_name=$run_name \
        --output_dir=$output_dir \
        --model_name=mixloradsi \
        --pretrained_path=$model_dir \
        --query_to_docid_path=$query_to_docid_path \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --save_safetensors=False \
        --save_total_limit=1 \
        --per_device_train_batch_size=512 \
        --learning_rate=1e-3 \
        --optim=adamw_8bit \
        --dataloader_num_workers=4 \
        --gradient_accumulation_steps=2 \
        --seed=42 \
        --bf16=True \
        --tf32=True \
        --num_train_epochs=8 \
        --save_steps=1000 \
        --weight_decay=0.01 \
        --warmup_ratio=0.045 \
        --logging_steps=1000 \
        --lr_scheduler_type=linear \
        --mixlora_config_json_path ./MixLoraDSI/longeval/config/mixloradsi_pretrained_d0.json \
        --report_to=none \
        --torch_empty_cache_steps=100