# Run as: bash examples/models/qwen25vl_satvidr1_FT_frozenvision.sh from lmms-eval root

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline" 

accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/jupyter/checkpoints/sat_vidr1_sft_frozenvis/2025-10-09_15-15-08/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks vsibench,blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,erqa,stare,sat_real \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=qwen25_satvidr1_sft,name=checkpoint-24000

#blink_reldepth,blink_sprel,blink_mv,blink_jigsaw