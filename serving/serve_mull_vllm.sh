#!/bin/bash
# Serve the Mull stage-2 checkpoint with vLLM's native Qwen2.5-VL implementation.
#
# Prereq: a vLLM-ready copy of the checkpoint (fixes image_processor_type and
# pins vision_config; see serving/prepare_for_vllm.py):
#
#   python serving/prepare_for_vllm.py --out /path/to/Qwen2.5-VL-Mull-vllm
#
# Env: recent vLLM + stock transformers. NOT the mull_tokens training env,
# whose transformers is the Video-R1 fork.
set -euo pipefail

MODEL=${MODEL:-/projectnb/ivc-ml/array/research/visual_reasoning/mull_analysis/Qwen2.5-VL-Mull-vllm}
PORT=${PORT:-8000}
TP=${TP:-1}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --chat-template supplies the Mull prompt: an assistant turn pre-filled with
# "<think>" + 20 * "<|latent_pad|>" + "</think>". Without it the model is served
# as a plain SFT Qwen2.5-VL and never gets its latent thinking tokens.
# Override the count per request with {"chat_template_kwargs": {"num_latents": N}}.
#
# No --trust-remote-code: the published auto_map points at
# mmlatentdiscrete_qwen_vl.py, which only imports under the transformers fork.
vllm serve "$MODEL" \
    --served-model-name Qwen2.5-VL-Mull \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --chat-template "$HERE/mull_chat_template.jinja" \
    --limit-mm-per-prompt '{"image": 8, "video": 1}' \
    --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 12845056}' \
    --generation-config auto
