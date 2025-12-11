# needs to be run from root
# bash google_scripts/launch_scripts/run_grpo_mmlatent1cont.sh

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

export WANDB_MODE="offline"

# set the exp config file
EXP_CONFIG_FILE="../../google_scripts/exp_configs/vidr1_sat_zebra_grpo_mmlatent2_qwenlatent1cont.yaml"

# For resume training:  --resume_from_checkpoint Model_Path \

# Qwen/Qwen2.5-VL-7B-Instruct

cd src/r1-v && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --exp_conf_file $EXP_CONFIG_FILE  \
    --output_dir "" \
    --model_name_or_path '' \
    --dataset_name "" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing false \
    --temporal false \
    --len_control false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name "" \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 2  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
