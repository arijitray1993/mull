# Run as: sh examples/models/vidr1_sat_zebra_grpo_qwenlatent2discrete_100.sh
# Configuration
export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="online" 

# --- LMMS-Eval Execution ---

# Define the model arguments, using the FULL_LOCAL_PATH
MODEL_ARGS="pretrained=array/Qwen2.5-VL-MullGRPO,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False"

echo "Starting LMMS Evaluation..."

# Get GPU IDs assigned by SLURM
# GPU_IDS=${CUDA_VISIBLE_DEVICES:-0}

accelerate launch --num_processes=3 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args="${MODEL_ARGS}" \
    --gen_kwargs=prompt_mode=mmlatent2,prompt_version=new,num_latents=20 \
    --tasks sitebench \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_sat_zebra_grpo_qwenlatent2,name=checkpoint-900 \
    --limit 2000


# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,sat_real,erqa,stare,vsibench,sitebench