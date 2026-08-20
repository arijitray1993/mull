# Serving Mull-Tokens with vLLM

The stage-2 checkpoint ([`array/Qwen2.5-VL-Mull`](https://huggingface.co/array/Qwen2.5-VL-Mull))
is architecturally a **stock Qwen2.5-VL-7B**: the checkpoint contains no weights
outside the standard name set, and the Mull-specific code paths in
`models/mmlatentdiscrete_qwen_vl.py` (the `latent_hidden_states` scatter and the
`pixel_values_latent` vision path) are gated on `not generate_mode`, i.e. they
only run during training. At inference `<|latent_pad|>` is an ordinary token
embedded from `embed_tokens`. So vLLM serves it with its native Qwen2.5-VL
implementation and **no custom model code**.

What it does need: two config fixes and one chat template.

> Only **stage 2** (and the GRPO checkpoint built on it) can be served this way.
> Stage-1 checkpoints feed continuous hidden states back into the embedding
> stream and would need a custom vLLM model plugin.

## 1. Build a vLLM-ready copy

```bash
python serving/prepare_for_vllm.py --out /path/to/Qwen2.5-VL-Mull-vllm
```

Weights are symlinked out of the HF cache, so the copy costs ~5 MB. It applies:

| Fix | Why |
|---|---|
| `preprocessor_config.json`: `image_processor_type` → `Qwen2VLImageProcessor` | The published value, `Qwen2_5_VLImageProcessor`, is the transformers-fork class name and is not in transformers' auto map — stock transformers ≥ 4.5x raises `Unrecognized image processor`. Qwen2.5-VL has always reused `Qwen2VLImageProcessor` (as `Qwen/Qwen2.5-VL-7B-Instruct` does); the two classes produce bit-identical `pixel_values`. |
| `config.json`: full explicit `vision_config` | The published config omits `depth`, `num_heads`, `intermediate_size`, `out_hidden_size`, `patch_size`, `spatial_merge_size`, `window_size`, `fullatt_block_indexes`, `temporal_patch_size` and relies on transformers' class defaults. They match Qwen2.5-VL-7B today, but that is not a contract. |
| `config.json`: drop `auto_map` | It points at `mmlatentdiscrete_qwen_vl.py`, which only imports under the Video-R1 transformers fork. Never serve this model with `--trust-remote-code`. |

Everything else in the published config is already correct for vLLM:
`vocab_size` 151669 (the 4 added latent tokens), `tie_word_embeddings: false`
with a real `lm_head.weight`, `rope_scaling.mrope_section` (so vLLM enables
M-RoPE), and pre-4.52 flat weight names, which vLLM's `hf_to_vllm_mapper`
handles explicitly.

## 2. Serve it

```bash
MODEL=/path/to/Qwen2.5-VL-Mull-vllm bash serving/serve_mull_vllm.sh
```

The one thing that is *not* optional is `--chat-template
serving/mull_chat_template.jinja`. Mull expects the assistant turn to be
pre-filled with the latent block:

```
<|im_start|>assistant
<think><|latent_pad|> x20</think>
```

Served with the checkpoint's own chat template, the model gets **zero latent
tokens** and behaves like a plain SFT Qwen2.5-VL. The template reproduces the
prompt built by `lmms_eval/models/chat/qwen2_5_vl_mmlatentdiscrete.py`
byte-for-byte, including its `text.replace("<|im_end|>\n", "")` — which strips
the terminator from *every* turn, not just the assistant prefill. That differs
from the training format in `dataloaders/custom_datasets.py`, but it is what the
reported numbers were produced with.

Client side, nothing is special — the latent block is injected by the template:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
client.chat.completions.create(
    model="Qwen2.5-VL-Mull",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": question},
    ]}],
    temperature=0, max_tokens=512,
    extra_body={"chat_template_kwargs": {"num_latents": 20}},  # optional, default 20
)
```

## 3. Matching the paper's eval settings

- **`max_pixels`** — the checkpoint's `preprocessor_config.json` defaults to
  401408, but the evals run `max_pixels=12845056`. Pass it via
  `--mm-processor-kwargs`, or accuracy will differ for reasons unrelated to vLLM.
- **`repetition_penalty`** — `generation_config.json` ships 1.05, which HF
  `generate()` applies silently. Keep `--generation-config auto` (vLLM's
  default), or set it explicitly per request.
- **Slow vs fast image processor** — transformers' fast processor differs from
  the slow one in the last significant digit of `pixel_values` (verified: sums
  −2753020.25 vs −2753019.25 on a test image). Harmless, but it is a source of
  occasional token-level divergence from an HF reference run.
- **Video** — `qwen2_5_vl_mmlatentdiscrete.py` subsamples frames with `linspace`
  and appends the last frame. vLLM's processor does not do this; pre-sample
  frames client-side to match.

## 4. Verifying a deployment

`serving/check_vllm_parity.py` runs the same SAT samples through the HF
reference path (the fork's `mmlatentdiscrete_qwen_vl` class, as lmms-eval loads
it) and through vLLM, then diffs prompt token ids, generated text, and answers:

```bash
python serving/check_vllm_parity.py prep --work-dir WORK --num-samples 24
<fork env>/python serving/check_vllm_parity.py run --backend hf   --work-dir WORK
<vllm env>/python serving/check_vllm_parity.py run --backend vllm --work-dir WORK
python serving/check_vllm_parity.py compare --work-dir WORK
```

The two backends need separate environments: the HF path requires the Video-R1
transformers fork, vLLM requires stock transformers.
