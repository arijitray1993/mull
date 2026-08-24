"""Produce the files to upload to array/Qwen2.5-VL-Mull so it serves correctly
out of the box, and report the exact diff.

    python serving/patch_hf_repo.py --out /path/to/staging

Writes only the files that change. Nothing is uploaded; print the upload command
and run it yourself when you are happy with the diff.
"""
import argparse
import difflib
import json
import os

from huggingface_hub import snapshot_download

from prepare_for_vllm import VISION_CONFIG  # single source of truth

NUM_LATENTS = 20

# The checkpoint's template ends by opening an empty assistant turn. Mull needs
# that turn pre-filled with the latent block, otherwise the model never receives
# a single <|latent_pad|> and reverts to thinking in text.
OLD_TAIL = "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
NEW_TAIL = (
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% set n = num_latents if num_latents is defined else " + str(NUM_LATENTS) + " %}"
    "{% if n > 0 %}<think>{{ '<|latent_pad|>' * n }}</think>\n{% endif %}"
    "{% endif %}"
)


def show_diff(name, old, new):
    if old == new:
        print(f"  {name}: unchanged")
        return False
    d = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                  fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="", n=1))
    print("\n".join("  " + l for l in d[:40]))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="array/Qwen2.5-VL-Mull")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-latents", type=int, default=NUM_LATENTS)
    args = ap.parse_args()

    src = snapshot_download(args.repo, allow_patterns=["*.json", "*.jinja"])
    os.makedirs(args.out, exist_ok=True)
    changed = []

    # 1. image_processor_type: the published name is the transformers-fork class
    #    and stock transformers cannot resolve it.
    prep = json.load(open(os.path.join(src, "preprocessor_config.json")))
    old = json.dumps(prep, indent=2, sort_keys=True)
    prep["image_processor_type"] = "Qwen2VLImageProcessor"
    new = json.dumps(prep, indent=2, sort_keys=True)
    print("preprocessor_config.json")
    if show_diff("preprocessor_config.json", old, new):
        open(os.path.join(args.out, "preprocessor_config.json"), "w").write(new + "\n")
        changed.append("preprocessor_config.json")

    # 2. chat template: pre-fill the assistant turn with the latent block.
    #    Guarded by add_generation_prompt, so training-style calls that pass an
    #    explicit assistant message are unaffected.
    tpl = open(os.path.join(src, "chat_template.jinja")).read()
    assert OLD_TAIL in tpl, "chat template tail not found; template changed upstream?"
    new_tpl = tpl.replace(OLD_TAIL, NEW_TAIL)
    print("\nchat_template.jinja (and chat_template.json, kept in sync)")
    if show_diff("chat_template.jinja", tpl, new_tpl):
        open(os.path.join(args.out, "chat_template.jinja"), "w").write(new_tpl)
        json.dump({"chat_template": new_tpl},
                  open(os.path.join(args.out, "chat_template.json"), "w"), indent=2)
        changed += ["chat_template.jinja", "chat_template.json"]

    # 3. vision_config written out explicitly instead of leaning on class defaults.
    #    auto_map is deliberately KEPT: vLLM ignores it without --trust-remote-code
    #    (verified on GPU), and removing it would break fork users who rely on it.
    cfg = json.load(open(os.path.join(src, "config.json")))
    old = json.dumps(cfg, indent=2, sort_keys=True)
    cfg["vision_config"] = VISION_CONFIG
    new = json.dumps(cfg, indent=2, sort_keys=True)
    print("\nconfig.json")
    if show_diff("config.json", old, new):
        open(os.path.join(args.out, "config.json"), "w").write(new + "\n")
        changed.append("config.json")

    print(f"\nstaged in {args.out}: {', '.join(changed)}")
    print("\nupload with:")
    for f in changed:
        print(f"  hf upload {args.repo} {os.path.join(args.out, f)} {f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
