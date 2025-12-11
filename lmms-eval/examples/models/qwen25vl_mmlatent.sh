# Run as: bash examples/models/qwen25vl_mmlatent.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline"

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatent \
    --model_args=pretrained=/home/jupyter/checkpoints/sat_zebra_sft_mmlatent/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_mode=mmlatent2 \
    --tasks mmvu \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_sat_zebra_sft_mmlatent,name=checkpoint_24000


# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench