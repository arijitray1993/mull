"""Build a vLLM-ready copy of the Mull stage-2 checkpoint.

The published config.json omits most of vision_config and relies on
transformers' class defaults to fill it in.  They happen to match
Qwen2.5-VL-7B-Instruct today, but that is luck, not a contract -- so we write
every field explicitly.  We also drop `auto_map`, which points at
mmlatentdiscrete_qwen_vl.py and only imports under the Video-R1 transformers
fork; vLLM resolves the architecture natively and must not load that file.

preprocessor_config.json needs a second fix: it names the image processor
"Qwen2_5_VLImageProcessor", which is the fork's class name and is not in
transformers' auto map (Qwen2.5-VL has always reused Qwen2VLImageProcessor, as
Qwen/Qwen2.5-VL-7B-Instruct's own config shows).  Stock transformers >= 4.5x
fails with "Unrecognized image processor" until it is renamed.

Weights are symlinked out of the HF cache, so the served copy costs ~5 MB.

    python serving/prepare_for_vllm.py --out /path/to/Qwen2.5-VL-Mull-vllm
"""
import argparse
import json
import os

from huggingface_hub import snapshot_download

# Explicit vision_config. Values are Qwen2.5-VL-7B-Instruct's, with the two
# fields the Mull config already pinned (hidden_size 1280, tokens_per_second 2).
VISION_CONFIG = {
    "model_type": "qwen2_5_vl",
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "window_size": 112,
    "fullatt_block_indexes": [7, 15, 23, 31],
    "tokens_per_second": 2,
    "initializer_range": 0.02,
    "torch_dtype": "bfloat16",
    "dtype": "bfloat16",
}

# Everything except the weight shards; shards get symlinked.
SMALL_FILES = [
    "*.json", "*.jinja", "*.txt", "merges.txt", "vocab.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="array/Qwen2.5-VL-Mull")
    ap.add_argument("--out", required=True, help="directory to write the vLLM-ready copy into")
    ap.add_argument("--weights", choices=["symlink", "copy", "skip"], default="symlink")
    args = ap.parse_args()

    patterns = list(SMALL_FILES) + (["*.safetensors"] if args.weights != "skip" else [])
    src = snapshot_download(args.repo, allow_patterns=patterns)
    os.makedirs(args.out, exist_ok=True)

    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(args.out, name)
        if name == "config.json" or name.endswith(".py"):
            continue  # rewritten below / deliberately dropped
        if os.path.lexists(d):
            os.remove(d)
        if name.endswith(".safetensors") and args.weights == "symlink":
            os.symlink(os.path.realpath(s), d)
        else:
            with open(s, "rb") as f_in, open(d, "wb") as f_out:
                f_out.write(f_in.read())

    # "Qwen2_5_VLImageProcessor" only exists in the transformers fork; stock
    # transformers cannot resolve it and AutoProcessor raises.
    prep_path = os.path.join(args.out, "preprocessor_config.json")
    prep = json.load(open(prep_path))
    old_ip = prep.get("image_processor_type")
    if old_ip != "Qwen2VLImageProcessor":
        prep["image_processor_type"] = "Qwen2VLImageProcessor"
        json.dump(prep, open(prep_path, "w"), indent=2, sort_keys=True)

    config = json.load(open(os.path.join(src, "config.json")))
    assert config.get("stage") == "stage2", (
        f"expected a stage-2 checkpoint, got stage={config.get('stage')!r}. "
        "Stage-1 checkpoints feed continuous hidden states back into the embedding "
        "stream and cannot be served by stock vLLM."
    )
    dropped = config.pop("auto_map", None)
    config["vision_config"] = VISION_CONFIG
    config["_name_or_path"] = args.repo
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2, sort_keys=True)

    print(f"source snapshot : {src}")
    print(f"served copy     : {args.out}")
    print(f"dropped auto_map: {dropped}")
    print(f"image_processor : {old_ip} -> {prep['image_processor_type']}")
    print(f"vision_config   : {len(VISION_CONFIG)} fields written explicitly")


if __name__ == "__main__":
    main()
