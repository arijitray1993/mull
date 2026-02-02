# Claude Setup Guide

This document helps AI assistants (like Claude) set up the Mull-Tokens repository for users.

## Quick Setup Checklist

When helping a user set up this repository, follow these steps:

1. **Check system requirements:**
   - CUDA 12.x
   - Python 3.10+
   - 6-8 GPUs with 40GB+ VRAM (for training)

2. **Install dependencies:**
   ```bash
   pip install -r requirements/requirements.txt
   ```

3. **Install Flash Attention:**
   - Check GLIBC version: `ldd --version`
   - If >= 2.32: `pip install flash-attn --no-build-isolation`
   - If < 2.32: Use prebuilt wheel (see Installation section in README.md)

4. **Install custom transformers:**
   ```bash
   git clone https://github.com/arijitray1993/Video-R1
   pip install -e Video-R1/
   ```

5. **Configure datasets** - For each dataset, either:
   - Use HuggingFace path (recommended, no setup needed)
   - Or download and update paths in config files

6. **Update config paths:**
   - Training configs: `google_scripts/exp_configs/*.yaml`
   - Eval configs: `lmms-eval/lmms_eval/tasks/*/*.yaml`

## Path Configuration Template

Create a `.env` or update configs with these paths:

```bash
# Training datasets
SAT_PATH="array/SAT"                           # HF or local
VIDEO_R1_PATH="/path/to/Video-R1-COT-165k.json"
ZEBRACOT_PATH="multimodal-reasoning-lab/Zebra-CoT"  # HF or local

# Evaluation datasets
BLINK_PATH="/path/to/BLINK"
VSIBENCH_PATH="/path/to/VSI-Bench"
ERQA_PATH="/path/to/ERQA"
SAT_EVAL_PATH="/path/to/SAT"  # Same as training or separate
# MMSI-Bench: No setup needed (HuggingFace)

# Checkpoints
CHECKPOINT_DIR="/path/to/checkpoints"
```

## Common Issues

1. **Flash Attention errors:** Ensure CUDA_HOME is set correctly
2. **OOM during training:** Reduce batch size or use gradient checkpointing
3. **Dataset not found:** Check if using HF path vs local path correctly
4. **GRPO training slow:** Reduce `num_generations` parameter

## Minimal Evaluation Setup

To run evaluation only (no training):

```bash
# 1. Install requirements
pip install -r requirements/requirements.txt
pip install flash-attn --no-build-isolation

# 2. Install custom transformers
git clone https://github.com/arijitray1993/Video-R1 && pip install -e Video-R1/

# 3. Run evaluation with HF models (no local checkpoints needed)
cd lmms-eval
accelerate launch --num_processes=4 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args="pretrained=array/Qwen2.5-VL-Mull,max_pixels=12845056,attn_implementation=flash_attention_2" \
    --gen_kwargs=prompt_mode=mmlatent2,num_latents=20 \
    --tasks mmsi_bench \
    --batch_size 1
```

## Training Pipeline Overview

The full training pipeline consists of 4 stages:

### Stage 0: Simple SFT (Baseline)
- Script: `bash google_scripts/launch_scripts/run_sat_vidr1_zebra_sft.sh`
- Config: `google_scripts/exp_configs/sat_vidr1_zebra_sft.yaml`
- Purpose: Standard fine-tuning without Mull-tokens

### Stage 1: Latent Compression
- Script: `bash google_scripts/launch_scripts/run_vidr1_mmlatent1_qwenbase.sh`
- Config: `google_scripts/exp_configs/vidr1_mmlatent1_qwenbase.yaml`
- Purpose: Learn to compress visual embeddings into latent tokens

### Stage 2: Discrete Latent Learning
- Script: `bash google_scripts/launch_scripts/run_sft_qwenlatent1_vidr1_SAT_zebra_mmlatent_stage2discrete.sh`
- Config: `google_scripts/exp_configs/vidr1_sat_zebra_sft_mmlatent2discrete_qwenlatent1.yaml`
- Requires: Stage 1 checkpoint (update `model_path` in config)

### Stage 3: GRPO (Reinforcement Learning)
- Script: `bash google_scripts/launch_scripts/run_grpo_sat_vidr1_zebra_qwenlatent2discrete_1_100.sh`
- Config: `google_scripts/exp_configs/vidr1_sat_zebra_grpo_mmlatent2discrete_qwenlatent1_100.yaml`
- Requires: Stage 2 checkpoint (update `model_path` in config)

## Key Config Parameters

When modifying configs for users, these are the most important fields:

```yaml
# Dataset paths (MUST be updated for user's environment)
train_dataset_args:
  video_r1_location: '/path/to/Video-R1-COT-165k.json'
  sat_location: 'array/SAT'  # Can use HF path
  zebracot_location: 'multimodal-reasoning-lab/Zebra-CoT'  # Can use HF path

# Model checkpoint (update for Stage 2 and GRPO)
model_path: '/path/to/previous/stage/checkpoint'

# Training mode flags
mmlatent_mode_stage1: True   # For Stage 1
mmlatent_mode_stage2: True   # For Stage 2 and GRPO
mmlatent_rl_mode: True       # For GRPO only
```

## Available Evaluation Tasks

| Task | Command Flag |
|------|--------------|
| BLINK IQ Test | `blink_iqtest` |
| BLINK Spatial Relations | `blink_sprel` |
| BLINK Multiple Views | `blink_mv` |
| BLINK Relative Depth | `blink_reldepth` |
| BLINK Jigsaw | `blink_jigsaw` |
| SAT | `sat_real` |
| VSI-Bench | `vsibench` |
| ERQA | `erqa` |
| MMSI-Bench | `mmsi_bench` |

Run multiple tasks: `--tasks blink_iqtest,sat_real,vsibench`
