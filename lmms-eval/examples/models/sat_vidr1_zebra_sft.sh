# Run as: bash examples/models/sat_vidr1_zebra_sft.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="online" 

accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=array/Qwen2.5-VL-SATBase,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks sitebench \
    --batch_size 1 \
    --wandb_args project=qwen25_satvidr1zebra_sft,name=checkpoint-24000 \
    --limit 3000

#blink_reldepth,blink_sprel,blink_mv,blink_jigsaw,vsibench,mmsi_bench,sat_real,stare,erqa