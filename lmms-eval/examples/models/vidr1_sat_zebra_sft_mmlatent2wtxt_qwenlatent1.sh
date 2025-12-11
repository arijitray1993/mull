# Run as: bash examples/models/vidr1_sat_zebra_sft_mmlatent2wtxt_qwenlatent1.sh
# from lmms-eval root directory

export HF_HOME="~/.cache/huggingface"
export WANDB_MODE="offline"

export CUDA_VISIBLE_DEVICES=0 && accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmlatent \
    --model_args=pretrained=/home/jupyter/checkpoints/vidr1_sat_zebra_sft_mmlatent2wtxt_qwenlatent1/2025-10-10_22-55-07/checkpoint-40000,max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False \
    --gen_kwargs=prompt_mode=mmlatent2,max_new_tokens=768 \
    --tasks stare \
    --batch_size 1 \
    --output_path "/home/jupyter/vis/" \
    --log_samples \
    --wandb_args "project=vidr1_sat_zebra_sft_mmlatent2wtxt_qwenlatent1,name=checkpoint_40000" \
    --limit 50


# blink_sprel,blink_mv,blink_reldepth,blink_jigsaw,vsibench 