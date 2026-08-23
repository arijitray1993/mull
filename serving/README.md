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

## 0. Fixing the checkpoint itself (recommended)

`serving/patch_hf_repo.py` stages the four files that make
`vllm serve array/Qwen2.5-VL-Mull` work with **no flags at all**, and prints the
diff plus the upload commands. It uploads nothing.

```bash
python serving/patch_hf_repo.py --out /path/to/staging
```

| file | change |
|---|---|
| `preprocessor_config.json` | `image_processor_type` -> `Qwen2VLImageProcessor` |
| `chat_template.jinja` + `.json` | the `add_generation_prompt` branch pre-fills `<think>` + 20 x `<|latent_pad|>` + `</think>` |
| `config.json` | `vision_config` written out explicitly (`auto_map` deliberately kept) |

The template change is guarded by `add_generation_prompt`, so calls that pass an
explicit assistant message -- training (`dataloaders/custom_datasets.py`) and
lmms-eval -- render byte-identically to today. `num_latents` is overridable per
request (`chat_template_kwargs`), and `num_latents=0` gives a plain assistant
turn for non-latent ablations.

If you would rather not touch the checkpoint, sections 1-3 do the same thing
with local files and server flags.

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

## 4. Measured results

24 SAT test items (shuffled choices, A=13/B=11), one L40S, vLLM 0.11.2 vs the
HF fork path, `max_tokens=256`. `prompt=` counts prompts whose token ids are
byte-identical to the reference; `text=` identical generations.

Matched preprocessing (401408 px both sides):

| variant | prompt= | text= | answer= | acc | latent | img tok |
|---|---|---|---|---|---|---|
| `hf_deflt` (reference) | 24/24 | 24/24 | 24/24 | 20/24 | 20 | 988 |
| vLLM, minimal fix, **no chat template** | 0/24 | 0/24 | 19/24 | 17/24 | **0** | 988 |
| vLLM, minimal fix, **with** the template | 24/24 | 23/24 | 23/24 | 19/24 | 20 | 988 |

Matched preprocessing (12845056 px both sides):

| variant | prompt= | text= | answer= | acc | latent | img tok |
|---|---|---|---|---|---|---|
| `hf_full` (reference) | 24/24 | 24/24 | 24/24 | 22/24 | 20 | 31104 |
| vLLM, full recipe | 24/24 | 23/24 | 23/24 | 21/24 | 20 | 31104 |

**vLLM reproduces the HF path.** Prompt token ids are identical 24/24 whenever
both sides use the same preprocessing; generations agree 23/24, the single
divergence being the expected bf16 / fast-image-processor noise.

**Serving without the chat template is a different model.** It loads and
answers, and the answers agree with the reference 19/24 — which is why it looks
fine in a demo — but the model gets zero latent tokens and falls back to
thinking in text:

| | mean output tokens | textual `THOUGHT` chains | no `<answer>` within 256 tok |
|---|---|---|---|
| with latent template | 8.9 | 0/24 | 0/24 |
| without | **101.2** | **24/24** | 2/24 |

That is ~11x the decode cost per request and an 8% truncation rate, on top of
bypassing the mechanism the checkpoint was trained for. Accuracy differences on
24 items are within noise; the behavioural difference is not.

## 5. Two traps that will silently corrupt a comparison

1. **The repo's vendored `src/qwen-vl-utils` sets `MAX_PIXELS = 256*28*28
   (200704)`**, against 12845056 in the installed `qwen-vl-utils` 0.0.14. If it
   lands on `sys.path` ahead of site-packages, images are downscaled ~64x
   (234 image tokens instead of 15552) and every comparison against vLLM is
   meaningless. `check_vllm_parity.py` asserts against this.
2. **`mull_tokens` (torch 2.5.1+cu124, flash-attn 2.5.9) cannot run on the
   Blackwell nodes** (`RTX PRO 6000`, sm_120): `CUDA error: no kernel image is
   available for execution on the device`. Pin `-l gpu_type=L40S` (or A100/A40/
   A6000) for anything using that env. The vLLM env (torch 2.9) is fine there.

## 6. Verifying a deployment

`serving/check_vllm_parity.py` runs the same SAT samples through the HF
reference path (the fork's `mmlatentdiscrete_qwen_vl` class, as lmms-eval loads
it) and through vLLM, then diffs prompt token ids, generated text, and answers:

```bash
python serving/check_vllm_parity.py prep --work-dir WORK --num-samples 24
<fork env>/python serving/check_vllm_parity.py run --backend hf --work-dir WORK \
    --tag hf_full --template mull --max-pixels 12845056
<vllm env>/python serving/check_vllm_parity.py run --backend vllm --work-dir WORK \
    --tag vllm_full --template mull --max-pixels 12845056
python serving/check_vllm_parity.py compare --work-dir WORK --ref hf_full --others vllm_full
```

`--template default` serves the checkpoint's own template (no latent tokens) and
`--max-pixels 0` uses the checkpoint's own 401408, so the table above is
reproducible variant by variant.

The two backends need separate environments: the HF path requires the Video-R1
transformers fork, vLLM requires stock transformers.
