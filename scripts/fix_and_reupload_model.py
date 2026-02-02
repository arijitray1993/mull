"""
Fix the model weights and re-upload to HuggingFace.

The issue: transformers ties embed_tokens and lm_head during loading even though
tie_word_embeddings=False. We need to save the model in a way that prevents this.
"""

import torch
import os
import shutil
from transformers import AutoModelForVision2Seq, AutoProcessor
from safetensors import safe_open
from huggingface_hub import hf_hub_download, HfApi
import json

model_name = "array/Qwen2.5-VL-MullGRPO"
output_dir = "/projectnb/ivc-ml/array/research/visual_reasoning/mull-tokens/fixed_model_upload"

print("="*70)
print("Fixing model: array/Qwen2.5-VL-MullGRPO")
print("="*70)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load model
print("\n[1/5] Loading model...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print(f"  Model loaded. Embeddings tied: {model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}")

# Step 2: Load the correct lm_head weights from checkpoint
print("\n[2/5] Loading correct lm_head weights from checkpoint...")
shard4 = hf_hub_download(model_name, "model-00004-of-00004.safetensors")
with safe_open(shard4, framework="pt") as f:
    correct_lm_head = f.get_tensor("lm_head.weight")

print(f"  Correct lm_head shape: {correct_lm_head.shape}")
print(f"  Correct lm_head[151665] norm: {correct_lm_head[151665].float().norm().item():.6f}")

# Step 3: Create new untied lm_head
print("\n[3/5] Creating untied lm_head...")
new_lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
new_lm_head.weight.data = correct_lm_head.to(torch.bfloat16)
model.lm_head = new_lm_head

print(f"  After fix - Embeddings tied: {model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()}")
print(f"  embed_tokens[151665] norm: {model.model.embed_tokens.weight[151665].float().norm().item():.6f}")
print(f"  lm_head[151665] norm: {model.lm_head.weight[151665].float().norm().item():.6f}")

# Step 4: Ensure config has tie_word_embeddings=False in ALL places
print("\n[4/5] Updating config...")
model.config.tie_word_embeddings = False

# CRITICAL: The issue is that get_text_config() returns text_config (a dict),
# and transformers uses getattr() which doesn't work on dicts.
# Solution: Remove text_config so get_text_config() falls back to main config
if hasattr(model.config, 'text_config') and model.config.text_config is not None:
    print(f"  Removing text_config (was type: {type(model.config.text_config).__name__})")
    model.config.text_config = None

print(f"  Main config tie_word_embeddings: {model.config.tie_word_embeddings}")

# Step 5: Save the fixed model
print("\n[5/5] Saving fixed model...")
model.save_pretrained(output_dir, safe_serialization=True)
processor.save_pretrained(output_dir)

# Copy the custom model code file
print("\n  Copying custom model code...")
custom_code_path = hf_hub_download(model_name, "mmlatentdiscrete_qwen_vl.py")
shutil.copy(custom_code_path, os.path.join(output_dir, "mmlatentdiscrete_qwen_vl.py"))

print(f"\n  Model saved to: {output_dir}")

# Verify the saved model
print("\n" + "="*70)
print("Verifying saved model...")
print("="*70)

with open(os.path.join(output_dir, "config.json")) as f:
    saved_config = json.load(f)
print(f"  Saved config tie_word_embeddings: {saved_config.get('tie_word_embeddings')}")

print("\n  Saved files:")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"    {f}: {size / 1e6:.1f} MB" if size > 1e6 else f"    {f}: {size / 1e3:.1f} KB")

# Check the saved weights
print("\n  Checking saved weights...")
index_path = os.path.join(output_dir, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    print(f"    embed_tokens location: {weight_map.get('model.embed_tokens.weight')}")
    print(f"    lm_head location: {weight_map.get('lm_head.weight')}")

print("\n" + "="*70)
print("Model fix complete! Now verifying it loads correctly...")
print("="*70)

# Verify by reloading
print("\n[Verification] Reloading saved model...")
del model
torch.cuda.empty_cache()

test_model = AutoModelForVision2Seq.from_pretrained(
    output_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

tied_after_reload = test_model.model.embed_tokens.weight.data_ptr() == test_model.lm_head.weight.data_ptr()
print(f"  Embeddings tied after reload: {tied_after_reload}")
print(f"  embed_tokens[151665] norm: {test_model.model.embed_tokens.weight[151665].float().norm().item():.6f}")
print(f"  lm_head[151665] norm: {test_model.lm_head.weight[151665].float().norm().item():.6f}")

if tied_after_reload:
    print("\n  WARNING: Embeddings are still tied after reload!")
    print("  The model save didn't fix the issue properly.")
else:
    print("\n  SUCCESS: Embeddings are NOT tied after reload!")
    print(f"\n  To upload to HuggingFace, run:")
    print(f"    huggingface-cli upload array/Qwen2.5-VL-MullGRPO {output_dir} .")
