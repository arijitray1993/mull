"""Minimal offline vLLM inference with the Mull stage-2 checkpoint.

    python serving/prepare_for_vllm.py --out /path/to/Qwen2.5-VL-Mull-vllm
    python serving/mull_vllm_offline.py --model /path/to/Qwen2.5-VL-Mull-vllm \
        --images photo.jpg --question "If you stand at the X and turn left, is the table left or right? A. left B. right"

The only Mull-specific part is the chat template: it pre-fills an assistant turn
with "<think>" + NUM_LATENTS * "<|latent_pad|>" + "</think>". Everything else is
stock Qwen2.5-VL -- vLLM needs no custom model code.
"""
import argparse
import os

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mull_chat_template.jinja")
NUM_LATENTS = 20

QUESTION_TEMPLATE = (
    "{Question}\nPlease think about this question deeply. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your final answer between the <answer> </answer> tags."
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    messages = [{"role": "user", "content": (
        [{"type": "image", "image": p} for p in args.images]
        + [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=args.question)}]
    )}]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=open(TEMPLATE).read(), num_latents=NUM_LATENTS,
    )

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=32768,
              limit_mm_per_prompt={"image": len(args.images)},
              mm_processor_kwargs={"min_pixels": 3136, "max_pixels": 12845056})
    # greedy, plus the repetition_penalty HF picks up from generation_config.json
    out = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {
            "image": [Image.open(p).convert("RGB") for p in args.images]}}],
        SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=args.max_tokens),
    )[0]

    print(f"latent tokens in prompt: {list(out.prompt_token_ids).count(151665)}")
    print(out.outputs[0].text)
