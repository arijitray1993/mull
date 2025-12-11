#!/bin/bash
# needs to be run from the Video-R1 root directory
# bash src/eval_bench.sh

model_paths=(
  "/home/jupyter/data_files/Qwen2.5-VL-7B-Instruct" # Qwen baseline
  "/home/jupyter/data_files/Qwen2.5-VL-7B-COT-SFT" # Video R1 trained baseline
  # "/home/jupyter/Video-R1/src/r1-v/log/sat_vidr1_sft/checkpoint-18000",
  # "/home/jupyter/checkpoints/sat_vidr1_sft_pause/checkpoint-8000"
)

export DECORD_EOF_RETRY_MAX=20480

CUDA_VISIBLE_DEVICES=0 python3 ./src/eval_bench.py \
  --model_path  \
  --file_name "$file_name" \
  --dataset_names "sat" "vsibench" "mmvu"
