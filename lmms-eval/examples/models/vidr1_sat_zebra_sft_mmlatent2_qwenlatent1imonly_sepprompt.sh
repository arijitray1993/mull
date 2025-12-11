# Run as: bash examples/models/vidr1_sat_zebra_sft_mmlatent2_qwenlatent1imonly_sepprompt.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatent_imonly \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_mmlatent2_mmlatent1imonly_sepprompt/2025-10-28_02-59-25/checkpoint-24000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_version=new_sepprompt,max_new_tokens=1024,temperature=0,do_sample=True \
    --tasks blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench,sat_real,stare,erqa \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_zebra_mmlatent2_mmlatent1imonly_sepprompt,name=checkpoint_24000


# /home/jupyter/checkpoints/vidr1_zebra_mmlatent1_qwenbase/offline-run-20250917_235322-w7x0nq1g/checkpoint-24000

# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench,sat_real,stare,erqa