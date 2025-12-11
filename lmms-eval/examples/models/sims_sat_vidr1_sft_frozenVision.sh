# Run as: bash examples/models/sims_sat_vidr1_sft_frozenVision.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline" 

accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/jupyter/checkpoints/sims_sat_vidr1_sft/2025-10-08_14-39-26/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,vsibench,sat_real,stare,erqa \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=sims_sat_vidr1_sft_frozenVision,name=checkpoint-24000

# blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,vsibench,sat_real,stare,erqa

# when reloading from gcs, use sims_vidr1_sft/ folder. 

