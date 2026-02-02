# Mull-Tokens: Multi-Modal Latent Tokens for Visual Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2512.10941-b31b1b.svg)](https://arxiv.org/abs/2512.10941)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://arijitray.com/multimodal_thinking/)

This repository contains the training and evaluation code for **Mull-Tokens**, a method that compresses visual information into discrete latent tokens for improved multi-modal reasoning with Qwen2.5-VL.

## Table of Contents

- [Installation](#installation)
- [Pre-trained Models](#pre-trained-models)
- [Minimal CLI Inference](#minimal-cli-inference)
- [Dataset Setup](#dataset-setup)
  - [Training Datasets](#training-datasets)
  - [Evaluation Datasets](#evaluation-datasets)
- [Training](#training)
  - [Simple SFT (Baseline)](#1-simple-sft-baseline)
  - [Mull-Tokens Stage 1](#2-mull-tokens-stage-1-latent-compression)
  - [Mull-Tokens Stage 2](#3-mull-tokens-stage-2-discrete-latent-learning)
  - [GRPO (Reinforcement Learning)](#4-grpo-reinforcement-learning)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)

---

## Installation

### Requirements

```bash
pip install -r requirements/requirements.txt
```

**Core dependencies:**
- `wandb` - Experiment tracking
- `deepspeed` - Distributed training
- `vllm` - Fast inference
- `qwen_vl_utils` - Vision processing
- `nltk`, `rouge_score` - Evaluation metrics

### Flash Attention

For systems with **GLIBC >= 2.32**:
```bash
pip install flash-attn --no-build-isolation
```

For systems with **GLIBC < 2.32** (e.g., RHEL/CentOS 8):
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### Custom Transformers

Install the custom transformers package with Mull-tokens vocabulary support:

```bash
git clone https://github.com/arijitray1993/Video-R1
cd Video-R1
pip install -e .
```

### Environment Setup

```bash
# CUDA setup (adjust for your system)
module load cuda/12.5
export CUDA_HOME=/path/to/cuda/12.5/install

# Optional: Configure HuggingFace cache
export HF_HOME="~/.cache/huggingface"

# Optional: WandB configuration
export WANDB_MODE="online"  # or "offline" for local logging
```

---

## Pre-trained Models

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| **Qwen2.5-VL-Mull** | Mull-Tokens Stage 2 (SFT) | [array/Qwen2.5-VL-Mull](https://huggingface.co/array/Qwen2.5-VL-Mull) |
| **Qwen2.5-VL-MullGRPO** | Mull-Tokens with GRPO | [array/Qwen2.5-VL-MullGRPO](https://huggingface.co/array/Qwen2.5-VL-MullGRPO) |

---

## Minimal CLI Inference

Quick inference with our pre-trained Mull-Tokens models.

### Basic Usage

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Choose model: "array/Qwen2.5-VL-Mull" or "array/Qwen2.5-VL-MullGRPO"
MODEL_ID = "array/Qwen2.5-VL-Mull"
NUM_LATENTS = 20

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Prepare input with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
    # IMPORTANT: Mull-Tokens requires latent thinking tokens before answer generation
    # Append as assistant message with "<think>" + "<|latent_pad|>"*20 + "</think>"
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "<think>" + "<|latent_pad|>" * NUM_LATENTS + "</think>\n" 
            }
        ],
    },
]

# Process inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
text = text.replace("<|im_end|>\n", "")  # Remove end token so model continues generating

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Generate response
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )

# Decode output (skip input tokens)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Multi-Image Input

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image1.jpg"},
            {"type": "image", "image": "image2.jpg"},
            {"type": "text", "text": "Compare these two images."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>" + "<|latent_pad|>" * 20 + "</think>\n"}],
    },
]
```

### Video Input

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/video.mp4", "max_pixels": 360*420, "fps": 1.0},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>" + "<|latent_pad|>" * 20 + "</think>\n"}],
    },
]
```


---


## Dataset Setup

### Training Datasets

#### 1. Video-R1

Video reasoning dataset with 165K chain-of-thought examples.

```bash
# Clone the Video-R1 repository
git clone https://github.com/tulerfeng/Video-R1
cd Video-R1

# Build environment
conda create -n video-r1 python=3.11
conda activate video-r1
bash setup.sh

# Qwen video extraction setting (max frames, resolutions)
# Use the [decord] feature to improve speed
cd src/qwen-vl-utils
pip install -e .[decord]
cd ..

# Download training dataset
git lfs install
git clone https://huggingface.co/datasets/Video-R1/Video-R1-data
```

Place the downloaded dataset in `/your_path/`, edit the path in `./src/unzip.py` in `root_directory` then unzip:

```bash
python ./src/unzip.py
```


### Evaluation Datasets

Recommended: Keep the online huggingface dataset links. 

If using an offline local setup, update the paths in `lmms-eval/lmms_eval/tasks/*/` YAML files to match your local setup.


## Training

All training scripts should be run from the repository root directory. 

### 1. Simple SFT (Baseline)

Standard supervised fine-tuning without Mull-tokens.

**Config:** `google_scripts/exp_configs/sat_vidr1_zebra_sft.yaml`

Update the config with your Video-R1 location:
```yaml
video_r1_location: '/path/to/Video-R1-COT-165k.json'
```

**Launch Script**
```bash
bash google_scripts/launch_scripts/run_sat_vidr1_zebra_sft.sh
```

**Key settings:**
- Dataset mix: SAT (60%), Video-R1 (20%), Zebra-CoT (20%)
- 6 GPUs, DeepSpeed Zero-2
- Learning rate: 1e-6
- 1 epoch

### 2. Mull-Tokens Stage 1 (Latent Compression)

Trains the model to compress visual embeddings into 20 discrete latent tokens.

**Config:** `google_scripts/exp_configs/vidr1_mmlatent1_qwenbase.yaml`

Update the config with your Video-R1 location:
```yaml
video_r1_location: '/path/to/Video-R1-COT-165k.json'
```

**Launch Script**
```bash
bash google_scripts/launch_scripts/run_vidr1_zebra_mmlatent1_qwenbase.sh
```


### 3. Mull-Tokens Stage 2 (Discrete Latent Learning)

Trains with discrete latent tokens from Stage 1.

**Config:** `google_scripts/exp_configs/vidr1_sat_zebra_sft_mmlatent2discrete_qwenlatent1.yaml`

**Prerequisites:**
- Completed Stage 1 checkpoint
- Update `model_path` in config to point to Stage 1 checkpoint

**Launch Script**
```bash
bash google_scripts/launch_scripts/run_sft_qwenlatent1_vidr1_SAT_zebra_mmlatent_stage2discrete.sh
```

### 4. GRPO (Reinforcement Learning)

Optimizes with Group Relative Policy Optimization.

**Config:** `google_scripts/exp_configs/vidr1_sat_zebra_grpo_mmlatent2discrete_qwenlatent1_new.yaml`

**Prerequisites:**
- Completed Stage 2 checkpoint
- Update `model_path` in config to point to Stage 2 checkpoint

**Launch Script**
```bash
bash google_scripts/launch_scripts/run_grpo_sat_vidr1_zebra_qwenlatent2discrete_1.sh
```

---

## Evaluation

Run evaluations using the lmms-eval framework.

### Quick Start

```bash
cd lmms-eval
sh examples/models/vidr1_sat_zebra_sft_mmlatent2discrete_qwenlatent1.sh
```

### Custom Evaluation

```bash
cd lmms-eval

MODEL_PATH="array/Qwen2.5-VL-Mull"  # or local checkpoint path
MODEL_ARGS="pretrained=${MODEL_PATH},max_pixels=12845056,max_num_frames=16,attn_implementation=flash_attention_2,interleave_visuals=False"

accelerate launch --num_processes=4 -m lmms_eval \
    --model qwen2_5_vl_mmlatentdiscrete \
    --model_args="${MODEL_ARGS}" \
    --gen_kwargs=prompt_mode=mmlatent2,num_latents=20 \
    --tasks blink_iqtest,blink_sprel,sat_real,vsibench,erqa,mmsi_bench \
    --batch_size 1 \
    --output_path "./eval_outputs"
```



## Configuration Reference

### Training Config Structure

```yaml
# Run identifier
run_name: 'experiment_name'

# Dataset configuration
train_dataset_args:
  split: train
  mix_datas:
    'SAT': 0.6          # Dataset weight (0-1)
    'VideoR1': 0.2
    'ZebraCOT': 0.2
  sat_location: 'array/SAT'                    # HF repo or local path
  video_r1_location: '/path/to/Video-R1.json'
  zebracot_location: 'multimodal-reasoning-lab/Zebra-CoT'
  mode: 'train'

  # Mull-tokens specific
  mmlatent_mode_stage1: False    # Enable for Stage 1
  mmlatent_mode_stage2: False    # Enable for Stage 2
  mmlatent_rl_mode: False        # Enable for GRPO
  num_latent_tokens: 20          # Number of latent tokens

# Model configuration
model_name: Qwen2.5-VL-7B        # or Qwen2.5-VL-7B-MMLatentDiscrete
model_path: 'path/to/model'

# Training options
freeze_vision: True              # Freeze vision encoder
latent_size: 20                  # THIS DEFUNCT AND NOT USED.
stage: stage1                    # stage1 or stage2
```


---

## Citation

If you use this code, please cite:

```bibtex
@misc{ray2025mulltokensmodalityagnosticlatentthinking,
      title={Mull-Tokens: Modality-Agnostic Latent Thinking}, 
      author={Arijit Ray and Ahmed Abdelkader and Chengzhi Mao and Bryan A. Plummer and Kate Saenko and Ranjay Krishna and Leonidas Guibas and Wen-Sheng Chu},
      year={2025},
      eprint={2512.10941},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10941}, 
}
```


## Acknowledgments

This work builds upon:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Video-R1](https://github.com/arijitray1993/Video-R1)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [MIRAGE] (https://github.com/UMass-Embodied-AGI/Mirage)
