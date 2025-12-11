# run from lmms-eval/ directory
# bash examples/models/qwen25vl_query.sh

export HF_HOME="~/.cache/huggingface"

export WANDB_MODE="offline"

accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_query \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_sft_query/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=True \
    --gen_kwargs=prompt_mode=query \
    --tasks blink_reldepth,blink_sprel,blink_mv \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_sat_sft_query,name=checkpoint_24000

# blink_reldepth,blink_sprel,blink_mv,blink_jigsaw