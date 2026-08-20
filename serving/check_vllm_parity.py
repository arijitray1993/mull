"""HF-vs-vLLM parity check for the Mull stage-2 checkpoint on SAT samples.

The two backends cannot share an environment (HF path needs the Video-R1
transformers fork, vLLM needs stock transformers >= 4.55), so run them
separately against the same prepared samples and diff the results:

    python serving/check_vllm_parity.py prep --work-dir WORK --num-samples 24
    <fork env>/python  serving/check_vllm_parity.py run --backend hf   --work-dir WORK
    <vllm env>/python  serving/check_vllm_parity.py run --backend vllm --work-dir WORK
    python serving/check_vllm_parity.py compare --work-dir WORK

Prompts come from serving/mull_chat_template.jinja for both backends, and that
template is byte-identical to what lmms_eval's qwen2_5_vl_mmlatentdiscrete
builds, so this exercises the real eval prompt.
"""
import argparse
import glob
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "mull_chat_template.jinja")
NUM_LATENTS = 20
LATENT_PAD_ID = 151665
MIN_PIXELS, MAX_PIXELS = 3136, 12845056

SAT_V2 = "/usr4/cs640g/array/.cache/huggingface/hub/datasets--array--SAT-v2/snapshots/*/data/test-*.parquet"

# verbatim from lmms-eval/lmms_eval/tasks/sat_real/utils.py (prompt_mode "latents")
QUESTION_TEMPLATE_LATENT = (
    "{Question}\n"
    "Please think about this question deeply. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
)
TYPE_TEMPLATE_MC = (
    " Please provide only the single option letter (e.g., A, B, C, D, etc.) "
    "within the <answer> </answer> tags."
)
IND_TO_LETTER = {0: "A", 1: "B", 2: "C"}


def render(tokenizer, messages, template="mull"):
    """template="mull": our latent-prefill template. "default": whatever the
    checkpoint ships, i.e. what `vllm serve` does with no --chat-template."""
    kw = {} if template == "default" else {
        "chat_template": open(TEMPLATE).read(), "num_latents": NUM_LATENTS}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kw)


def messages_for(sample):
    content = [{"type": "image", "image": p} for p in sample["images"]]
    content.append({"type": "text", "text": sample["prompt"]})
    return [{"role": "user", "content": content}]


def cmd_prep(args):
    import io
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted(glob.glob(SAT_V2))
    assert files, f"no SAT-v2 test parquet under {SAT_V2}"
    img_dir = os.path.join(args.work_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    samples = []
    for batch in pq.ParquetFile(files[0]).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if len(samples) >= args.num_samples:
                break
            if not 2 <= len(row["answers"]) <= 3:
                continue  # sat_real's letter map only covers A/B/C
            # SAT-v2 lists correct_answer first, so unshuffled samples are 100% "A"
            # and accuracy is meaningless (sat_real's process_docs works around this
            # by appending a reversed-answer copy of every doc). Shuffle per sample.
            answers = list(row["answers"])
            random.Random(f"{args.seed}-{len(samples)}").shuffle(answers)
            choices = "\n".join(f"({IND_TO_LETTER[i]}) {a}" for i, a in enumerate(answers))
            full = row["question"] + " Choose from the following options: \n" + choices
            paths = []
            for j, im in enumerate(row["images"]):
                p = os.path.join(img_dir, f"{len(samples):03d}_{j}.png")
                Image.open(io.BytesIO(im["bytes"])).convert("RGB").save(p)
                paths.append(p)
            samples.append({
                "images": paths,
                "prompt": QUESTION_TEMPLATE_LATENT.format(Question=full) + TYPE_TEMPLATE_MC,
                "gt": IND_TO_LETTER[answers.index(row["correct_answer"])],
                "question_type": row["question_type"],
            })
        if len(samples) >= args.num_samples:
            break

    json.dump(samples, open(os.path.join(args.work_dir, "samples.json"), "w"), indent=1)
    n_img = sum(len(s["images"]) for s in samples)
    dist = {l: sum(s["gt"] == l for s in samples) for l in "ABC"}
    print(f"wrote {len(samples)} samples ({n_img} images) to {args.work_dir}; gt distribution {dist}")


def run_hf(samples, model_path, max_tokens, template, max_pixels):
    import importlib
    import sys
    import torch
    from transformers import AutoProcessor

    sys.path.insert(0, os.path.join(REPO, "models"))
    # NB: import qwen_vl_utils from site-packages, NOT the repo's vendored
    # src/qwen-vl-utils (MAX_PIXELS = 256*28*28 there, which silently downscales
    # images ~64x and makes any comparison against vLLM meaningless).
    from qwen_vl_utils import process_vision_info
    import qwen_vl_utils.vision_process as _vp
    assert "/src/qwen-vl-utils/" not in _vp.__file__, f"vendored qwen_vl_utils on path: {_vp.__file__}"
    print(f"qwen_vl_utils: {_vp.__file__}", flush=True)

    # same class the eval harness imports, so this is the reference code path
    cls = importlib.import_module("mmlatentdiscrete_qwen_vl").Qwen2_5_VLForConditionalGeneration
    proc_kw = {"max_pixels": max_pixels, "min_pixels": MIN_PIXELS} if max_pixels else {}
    proc = AutoProcessor.from_pretrained(model_path, **proc_kw)
    model = cls.from_pretrained(
        model_path, torch_dtype="bfloat16", device_map="cuda",
        attn_implementation="flash_attention_2").eval()

    outs = []
    for s in samples:
        msgs = messages_for(s)
        text = render(proc.tokenizer, msgs, template)
        images, videos = process_vision_info(msgs)
        inputs = proc(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 max_new_tokens=max_tokens)
        outs.append({
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "text": proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0],
        })
        print(f"[hf {len(outs)}/{len(samples)}] {outs[-1]['text'][:80]!r}", flush=True)
    return outs


def run_vllm(samples, model_path, max_tokens, template, max_pixels):
    from PIL import Image
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    extra = ({"mm_processor_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": max_pixels}}
             if max_pixels else {})   # no max_pixels -> "no extra flags", checkpoint default
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=32768,
              limit_mm_per_prompt={"image": 8}, gpu_memory_utilization=0.85, **extra)
    # greedy, plus the repetition_penalty HF picks up from generation_config.json
    sp = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=max_tokens)

    reqs = [{
        "prompt": render(tokenizer, messages_for(s), template),
        "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in s["images"]]},
    } for s in samples]
    res = llm.generate(reqs, sp)
    return [{"prompt_ids": list(r.prompt_token_ids), "text": r.outputs[0].text} for r in res]


def cmd_run(args):
    samples = json.load(open(os.path.join(args.work_dir, "samples.json")))
    fn = run_hf if args.backend == "hf" else run_vllm
    outs = fn(samples, args.model, args.max_tokens, args.template, args.max_pixels)
    tag = args.tag or args.backend
    json.dump(outs, open(os.path.join(args.work_dir, f"{tag}.json"), "w"), indent=1)
    n_lat = outs[0]["prompt_ids"].count(LATENT_PAD_ID) if outs else 0
    n_img = outs[0]["prompt_ids"].count(151655) if outs else 0
    print(f"wrote {tag}.json ({len(outs)} rows) | prompt[0]: {len(outs[0]['prompt_ids'])} tokens, "
          f"{n_img} image, {n_lat} latent")


def extract(text):
    m = re.search(r"<answer>\s*([A-C])", text)
    return m.group(1) if m else None


def cmd_compare(args):
    samples = json.load(open(os.path.join(args.work_dir, "samples.json")))
    load = lambda t: json.load(open(os.path.join(args.work_dir, f"{t}.json")))
    ref = load(args.ref)
    n = len(samples)

    print(f"reference: {args.ref}  ({len(ref[0]['prompt_ids'])} prompt tokens, "
          f"{ref[0]['prompt_ids'].count(151655)} image, "
          f"{ref[0]['prompt_ids'].count(LATENT_PAD_ID)} latent)")
    print(f"{'variant':<22} {'prompt=':>8} {'text=':>7} {'answer=':>8} {'acc':>7}  latent  imgtok")
    print(f"{args.ref + ' (ref)':<22} {n:>7}/{n} {n:>6}/{n} {n:>7}/{n} "
          f"{sum(extract(r['text']) == s['gt'] for r, s in zip(ref, samples)):>4}/{n}  "
          f"{ref[0]['prompt_ids'].count(LATENT_PAD_ID):>6}  {ref[0]['prompt_ids'].count(151655):>6}")

    for tag in args.others:
        cur = load(tag)
        sp = sum(a["prompt_ids"] == b["prompt_ids"] for a, b in zip(ref, cur))
        st = sum(a["text"].strip() == b["text"].strip() for a, b in zip(ref, cur))
        sa = sum(extract(a["text"]) == extract(b["text"]) for a, b in zip(ref, cur))
        ac = sum(extract(b["text"]) == s["gt"] for b, s in zip(cur, samples))
        print(f"{tag:<22} {sp:>7}/{n} {st:>6}/{n} {sa:>7}/{n} {ac:>4}/{n}  "
              f"{cur[0]['prompt_ids'].count(LATENT_PAD_ID):>6}  {cur[0]['prompt_ids'].count(151655):>6}")

    if args.verbose:
        for tag in args.others:
            cur = load(tag)
            for i, (s, a, b) in enumerate(zip(samples, ref, cur)):
                if a["text"].strip() != b["text"].strip():
                    print(f"\n[{tag} {i:02d}] gt={s['gt']}\n  ref: {a['text'][:160]!r}\n  cur: {b['text'][:160]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_model = "/projectnb/ivc-ml/array/research/visual_reasoning/mull_analysis/Qwen2.5-VL-Mull-vllm"

    p = sub.add_parser("prep"); p.add_argument("--work-dir", required=True)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_prep)

    p = sub.add_parser("run"); p.add_argument("--work-dir", required=True)
    p.add_argument("--backend", choices=["hf", "vllm"], required=True)
    p.add_argument("--model", default=default_model)
    p.add_argument("--tag", help="output file name (default: the backend name)")
    p.add_argument("--template", choices=["mull", "default"], default="mull")
    p.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                   help="0 = do not override, use the checkpoint's own 401408")
    p.add_argument("--max-tokens", type=int, default=256); p.set_defaults(fn=cmd_run)

    p = sub.add_parser("compare"); p.add_argument("--work-dir", required=True)
    p.add_argument("--ref", default="hf")
    p.add_argument("--others", nargs="+", default=["vllm"])
    p.add_argument("--verbose", action="store_true"); p.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    args.fn(args)
