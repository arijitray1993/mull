# Run as: bash examples/models/vidr1_sat_zebra_grponoref_qwenlatent2.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline"

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatentsample \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_grponoref_mmlatent2_qwenlatent1/2025-10-14_14-32-15/checkpoint-100,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_mode=mmlatent2,prompt_version=new \
    --tasks erqa,stare \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=vidr1_sat_zebra_grponoref_qwenlatent2,name=checkpoint-100 \
    --limit 20


# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,sat_real,erqa,stare,vsibench 