# bash examples/models/qwen25vl_pause.sh
export HF_HOME="~/.cache/huggingface"

export WANDB_MODE="offline"

accelerate launch --num_processes=5 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_query \
    --model_args=pretrained=/home/jupyter/checkpoints/sat_vidr1_sft_pause/checkpoint-8000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks blink_reldepth,blink_sprel,blink_mv,blink_jigsaw \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=sat_vidr1_sft_pause,name=checkpoint_8000

