# Run as: bash examples/models/vidr1_sat_zebra_sft_mmlatent1imonly_qwenbase.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 && accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatent_imonly \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_mmlatent1imonly_qwenbase/2025-10-21_01-20-28/checkpoint-36000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_version=new,max_new_tokens=1024,temperature=0,do_sample=True \
    --tasks blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,erqa,stare,vsibench,sat_real \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_zebra_mmlatent1_qwenbase_2025-10-17_00-55-09,name=checkpoint_36000


# /home/jupyter/checkpoints/vidr1_zebra_mmlatent1_qwenbase/offline-run-20250917_235322-w7x0nq1g/checkpoint-24000

# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench,sat_real x