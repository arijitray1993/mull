# Run as: bash examples/models/qwen25vl.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline" 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 && accelerate launch --num_processes=6 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/jupyter/data_files/Qwen2.5-VL-7B-COT-SFT,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_version=new,max_new_tokens=1024 \
    --tasks blink_iqtest \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args project=videor1_baseline,name=baseline

# blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,sat_real,stare,erqa,vsibench
