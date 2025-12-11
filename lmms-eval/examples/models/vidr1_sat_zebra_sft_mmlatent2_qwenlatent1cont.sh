# Run as: bash examples/models/vidr1_sat_zebra_sft_mmlatent2_qwenlatent1cont.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline"

accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatent \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_sft_mmlatent2_qwenlatent1cont/2025-09-29_15-44-39/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_mode=mmlatent2 \
    --tasks blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench,erqa,stare,sat_real \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_sat_zebra_sft_mmlatent2_qwenlatent1cont,name=checkpoint_24000

# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench,erqa,stare,sat_real