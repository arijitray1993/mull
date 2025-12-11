# Run as: bash examples/models/sat_vidr1cot_zebraans_grpo.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline" 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 && accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_grpo_textCOT/2025-10-15_23-47-29/checkpoint-50,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_version=new,max_new_tokens=1024,num_latents=100 \
    --tasks blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,sat_real,stare,erqa,vsibench \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=sat_vidr1cot_zebraans_grpo,name=checkpoint-100

# blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,vsibench,sat_real,stare,erqa