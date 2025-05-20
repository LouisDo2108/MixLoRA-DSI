

source ~/.bashrc
conda activate mixlora_dsi
cd ./MixLoraDSI

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TQDM_DISABLE=1

# Array of experiment configurations
declare -a run_names=(
    "0.5-0.1"
    "0.5-0.5"
    "0.5-0.05"
)

declare -a initial_configs=(
    "./MixLoraDSI/config/ablations/alphas/mixloradsi/0.5-0.1.json"
    "./MixLoraDSI/config/ablations/alphas/mixloradsi/0.5-0.5.json"
    "./MixLoraDSI/config/ablations/alphas/mixloradsi/0.5-0.05.json"
)
root_dir=/home/thuy0050/ft49_scratch2/thuy0050/mixloradsi/nq320k

# Loop through the experiments
for ((exp=0; exp<${#run_names[@]}; exp++))
do
    echo "Running experiment with run_name: ${run_names[exp]}"
    
    # Set the current run's configuration
    run_name=${run_names[exp]}
    initial_mixlora_config_path=${initial_configs[exp]}
    
    # Your original script, with run_name and initial_mixlora_config_path replaced
    model_name=mixloradsi
    experiment_dir=experiments
    num_tasks=1
    
    for (( i = 1; i <= num_tasks; i++ ))
    do
        current_split=d$i
        previous_split=d$((i-1))
        data_root_dir=$root_dir/$current_split
        query_to_docid_path=$data_root_dir/doc2query/query_to_docid.train.json
        output_dir=$data_root_dir/$experiment_dir/ablation/alphas/$run_name
        docid_to_smtid_path=$data_root_dir/$experiment_dir/rq_smtid/docid_to_tokenids.json
        
        if [ $i -eq 1 ]; then
            model_dir=$root_dir/d0/$experiment_dir/mixloradsi
            mixlora_config_json_path=$initial_mixlora_config_path
            epoch=10
        else
            model_dir=$root_dir/$previous_split/$experiment_dir/ablation/alphas/$run_name
            mixlora_config_json_path=$root_dir/$previous_split/$experiment_dir/ablation/alphas/$run_name/mixlora_config.json
            epoch=10
        fi
        
        # printf "##### Training task $i #####\n\n"
        # python ./MixLoraDSI/train_mixloradsi.py \
        # --run_name=$current_split\_$run_name \
        # --output_dir=$output_dir \
        # --model_name=$model_name \
        # --pretrained_path=$model_dir \
        # --query_to_docid_path=$query_to_docid_path \
        # --docid_to_smtid_path=$docid_to_smtid_path \
        # --mixlora_config_json_path=$mixlora_config_json_path \
        # --taskid=$i \
        # --save_safetensors=False \
        # --save_total_limit=1 \
        # --learning_rate=1e-3 \
        # --optim=adamw_8bit \
        # --dataloader_num_workers=4 \
        # --gradient_accumulation_steps=1 \
        # --per_device_train_batch_size=32 \
        # --bf16=True \
        # --tf32=True \
        # --num_train_epochs=$epoch \
        # --weight_decay=0.01 \
        # --save_steps=2000 \
        # --warmup_ratio=0.1 \
        # --logging_steps=200 \
        # --lr_scheduler_type=linear \
        # --max_grad_norm=1.0 \
        # --report_to=none
        
        printf "##### Finished training task $i#####\n\n"
        printf "##### Evaluation after training task $i#####\n\n"
        
        for ((eval_i=0; eval_i<=$i; eval_i++))
        do
            printf "##### Evaluating task $eval_i with checkpoint trained upto task $i#####\n"
            train_split=$current_split
            data_root_dir=$root_dir/$train_split
            model_dir=$data_root_dir/$experiment_dir/ablation/alphas/$run_name
            out_dir=$data_root_dir/$experiment_dir/ablation/alphas/$run_name/out
            docid_to_tokenids_path=$root_dir/$train_split/experiments/rq_smtid/docid_to_tokenids.json
            eval_split=d$eval_i
            q_collection_paths=$root_dir/$eval_split/eval_queries
            eval_qrel_path=$q_collection_paths/eval_qrel.json
            mixlora_config_json_path=$root_dir/$train_split/$experiment_dir/ablation/alphas/$run_name/mixlora_config.json
            
            torchrun --master_port=25150 --nproc_per_node=1 eval_mixlora.py \
            --pretrained_path=$model_dir \
            --mixlora_config_json_path=$mixlora_config_json_path \
            --docid_to_tokenids_path=$docid_to_tokenids_path \
            --out_dir=$out_dir \
            --q_collection_paths=$q_collection_paths \
            --eval_qrel_path=$eval_qrel_path \
            --max_new_token_for_docid=8 \
            --batch_size 512 \
            --topk 10
            
            printf "\n\n"
        done
        
        printf "##### Finished evaluation after training task $i#####\n\n"
    done
    
    echo "Completed experiment with run_name: ${run_names[exp]}"
    echo "-----------------------------------"
done