# ==== ENVIRONMENT SETUP ====
source ~/.bashrc
conda activate mixlora_dsi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TQDM_DISABLE=1

cd ./MixLoraDSI

# ==== GLOBAL CONFIG ====
model_name="mixloradsi"
initial_checkpoint_name="mixloradsi-pt"
experiment_dir="experiments"
run_name="clever512"
base_dir="./mixloradsi/nq320k"
initial_config="./MixLoraDSI/config/baselines/clever.json"
epoch=5
num_tasks=4
batch_size=128 # We use batch size of 32 for NQ320k MixLoRA-DSI with pre-trained

# ==== TRAINING + EVALUATION LOOP ====
for ((i = 1; i <= num_tasks; i++)); do
  current_split="d$i"
  previous_split="d$((i - 1))"

  data_dir="$base_dir/$current_split"
  output_dir="$data_dir/$experiment_dir/baselines/$run_name"
  query_to_docid_path="$data_dir/doc2query/query_to_docid.train.json"
  docid_to_smtid_path="$data_dir/$experiment_dir/rq_smtid/docid_to_tokenids.json"

  if [ $i -eq 1 ]; then
    model_dir="$base_dir/d0/$experiment_dir/$initial_checkpoint_name"
    config_path="$initial_config"
  else
    model_dir="$base_dir/$previous_split/$experiment_dir/$run_name"
    config_path="$model_dir/mixlora_config.json"
  fi

  echo "##### Training task $i #####"
  # python train_clever.py \
  #   --run_name="${current_split}_${run_name}" \
  #   --output_dir="$output_dir" \
  #   --model_name="$model_name" \
  #   --pretrained_path="$model_dir" \
  #   --query_to_docid_path="$query_to_docid_path" \
  #   --docid_to_smtid_path="$docid_to_smtid_path" \
  #   --mixlora_config_json_path="$config_path" \
  #   --taskid="$i" \
  #   --save_safetensors=False \
  #   --save_total_limit=1 \
  #   --learning_rate=1e-3 \
  #   --optim=adamw_8bit \
  #   --dataloader_num_workers=4 \
  #   --gradient_accumulation_steps=1 \
  #   --per_device_train_batch_size=$batch_size \
  #   --bf16=True \
  #   --tf32=True \
  #   --num_train_epochs="$epoch" \
  #   --weight_decay=0.01 \
  #   --save_steps=1000 \
  #   --warmup_ratio=0.1 \
  #   --logging_steps=100 \
  #   --lr_scheduler_type=linear \
  #   --max_grad_norm=1.0 \
  #   --report_to=none \
  #   --dataset=nq320k

  echo "##### Finished training task $i #####"
  echo "##### Evaluation after training task $i #####"

  for ((eval_i = 0; eval_i <= i; eval_i++)); do
    echo "##### Evaluating task $eval_i with checkpoint from task $i #####"

    eval_split="d$eval_i"
    eval_query_dir="$base_dir/$eval_split/eval_queries"
    eval_qrel_path="$eval_query_dir/eval_qrel.json"

    eval_model_dir="$output_dir"
    out_dir="$eval_model_dir/out"
    config_path="$eval_model_dir/mixlora_config.json"

    python eval_mixlora.py \
      --pretrained_path="$eval_model_dir" \
      --mixlora_config_json_path="$config_path" \
      --docid_to_tokenids_path="$docid_to_smtid_path" \
      --out_dir="$out_dir" \
      --q_collection_paths="$eval_query_dir" \
      --eval_qrel_path="$eval_qrel_path" \
      --max_new_token_for_docid=8 \
      --batch_size=128 \
      --topk=10
    echo ""
  done

  echo "##### Finished evaluation for task $i #####"
  echo ""
done