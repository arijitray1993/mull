#!/bin/bash

# run as: cd lmms-eval && sh examples/models/vidr1_sat_zebra_sft_mmlatent2discrete_qwenlatent1.sh
# Configuration
# export CUDA_HOME=/share/pkg.8/cuda/12.2/installm
export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="online"

MODEL_PATH="array/Qwen2.5-VL-Mull"

# --- LMMS-Eval Execution ---

# Define the model arguments, using the FULL_LOCAL_PATH
MODEL_ARGS="pretrained=${MODEL_PATH},max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False"

echo "Starting LMMS Evaluation on VSI-Bench..."

OUTPUT_DIR="/projectnb/ivc-ml/array/research/visual_reasoning/mull-tokens/eval_outputs/vsibench"

accelerate launch --num_processes=2 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args="${MODEL_ARGS}" \
    --gen_kwargs=prompt_mode=mmlatent2,prompt_version=new,num_latents=20 \
    --tasks vsibench \
    --batch_size 1 \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    --wandb_args project=vidr1_sat_zebra_sft_mmlatent2_qwenlatent1,name="vsibench_eval"


# blink_iqtest,blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,erqa,sat_real,vsibench,stare
# 
# 

# /home/jupyter/checkpoints/vidr1_sat_zebra_sft_mmlatent2_qwenlatent1/2025-10-17_16-24-43/checkpoint-24000

# /home/jupyter/checkpoints/vidr1_sat_zebra_sft_mmlatent2_qwenlatent1/2025-09-19_00-11-43/checkpoint-24000

# /home/jupyter/checkpoints/vidr1_sat_zebra_sft_mmlatent2_qwenlatent1/2025-10-23_03-01-33/checkpoint-20000