# needs to be run from root
# bash google_scripts/launch_scripts/run_sims_sat_vidr1_sft.sh

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_MODE="offline"

# set the exp config file
EXP_CONFIG_FILE="../../google_scripts/exp_configs/sims_sat_vidr1_sft_unfrozenvis.yaml"

cd src/r1-v && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --exp_conf_file $EXP_CONFIG_FILE  \
    --output_dir "" \
    --model_name_or_path "" \
    --dataset_name "" \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name ""  \
    --save_steps 4000 \
    --max_grad_norm 5