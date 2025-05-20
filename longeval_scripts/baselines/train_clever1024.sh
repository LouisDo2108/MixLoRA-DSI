export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TQDM_DISABLE=1

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI


model_name=mixloradsi
experiment_dir=experiments
run_name=clever1024
num_tasks=3

for (( i = 3; i <= num_tasks; i++ ))
do

  current_split=d$i
  previous_split=d$((i-1))

  data_root_dir=./mixloradsi/longeval/$current_split
  query_to_docid_path=$data_root_dir/doc2query/query_to_docid.train.json # Only pseudo queries, **could merge with natural queries later**
  output_dir=$data_root_dir/$experiment_dir/$run_name
  docid_to_smtid_path=$data_root_dir/$experiment_dir/rq_smtid/docid_to_tokenids.json

  # This is done only once, since during indexing D1, CLEVER cannot sample from D0
  if [ $i -eq 1 ]; then 
    model_dir=./mixloradsi/longeval/d0/$experiment_dir/mixloradsi-pt
    mixlora_config_json_path=./MixLoraDSI/config/baselines/clever.json
    epoch=3
  else
    model_dir=./mixloradsi/longeval/$previous_split/$experiment_dir/$run_name
    mixlora_config_json_path=$model_dir/mixlora_config.json
    epoch=3
  fi
  
  printf "##### Training task $i #####\n\n"

  python train_clever_1024.py \
  --run_name=$current_split\_$run_name \
  --output_dir=$output_dir \
  --model_name=$model_name \
  --pretrained_path=$model_dir \
  --query_to_docid_path=$query_to_docid_path \
  --docid_to_smtid_path=$docid_to_smtid_path \
  --mixlora_config_json_path=$mixlora_config_json_path \
  --taskid=$i \
  --save_safetensors=False \
  --save_total_limit=1 \
  --learning_rate=1e-3 \
  --optim=adamw_8bit \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=2 \
  --per_device_train_batch_size=256 \
  --bf16=True \
  --tf32=True \
  --num_train_epochs=$epoch \
  --weight_decay=0.01 \
  --save_steps=1000 \
  --warmup_ratio=0.1 \
  --logging_steps=100 \
  --lr_scheduler_type=linear \
  --max_grad_norm=1.0 \
  --report_to=none \
  --dataset=longeval

  printf "##### Finished training task $i#####\n\n"

  printf "##### Evaluation after training task $i#####\n\n"
  for ((eval_i=0; eval_i<=$i; eval_i++))
  do
        printf "##### Evaluating task $eval_i with checkpoint trained upto task $i#####\n"

        train_split=$current_split
        data_root_dir=./mixloradsi/longeval/$train_split
        model_dir=$data_root_dir/$experiment_dir/$run_name
        out_dir=$data_root_dir/$experiment_dir/$run_name/out
        docid_to_tokenids_path=./mixloradsi/longeval/$train_split/experiments/rq_smtid/docid_to_tokenids.json
        
        eval_split=d$eval_i
        q_collection_paths=./mixloradsi/longeval/$eval_split/eval_queries
        eval_qrel_path=$q_collection_paths/eval_qrel.json
        mixlora_config_json_path=$model_dir/mixlora_config.json

        torchrun --master_port=25260 --nproc_per_node=1 eval_mixlora.py \
                --pretrained_path=$model_dir \
                --mixlora_config_json_path=$mixlora_config_json_path \
                --docid_to_tokenids_path=$docid_to_tokenids_path \
                --out_dir=$out_dir \
                --q_collection_paths=$q_collection_paths \
                --eval_qrel_path=$eval_qrel_path \
                --max_new_token_for_docid=8 \
                --batch_size 256 \
                --topk 10
        printf "\n\n"
  done
  printf "##### Finished evaluation after training task $i#####\n\n"
done