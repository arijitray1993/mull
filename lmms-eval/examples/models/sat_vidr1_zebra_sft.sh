# Run as: bash examples/models/sat_vidr1_zebra_sft.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline" 

accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/jupyter/checkpoints/sat_vidr1_zebra_sft/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks vsibench \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=qwen25_satvidr1zebra_sft,name=checkpoint-24000

#blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,vsibench