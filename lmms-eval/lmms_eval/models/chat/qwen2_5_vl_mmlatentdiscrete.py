# pylint: disable=all
import importlib
import os
import time
from typing import List, Optional, Tuple, Union
import pdb
import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
import json
import datetime
import io

from lmms_eval import utils
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

import sys

import base64
import re
from io import BytesIO

import decord
import torch
from accelerate import Accelerator, DistributedType
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

class LogExamples:
    def __init__(self, run_id: str, latent_token_id: int):
        """
        Initializes the logger.

        Args:
            run_id (str): A unique ID for this evaluation run.
            latent_token_id (int): The token ID for your latent_pad tokens.
        """
        self.run_id = run_id
        self.latent_token_id = latent_token_id
        self.global_counter = 0

        # Open a file to store the examples.
        # Using .jsonl (JSON Lines) as the extension is standard for this format.
        output_dir = "/projectnb/ivc-ml/array/research/visual_reasoning/mull-tokens/eval_outputs"
        os.makedirs(output_dir, exist_ok=True)
        filepath = f"{output_dir}/examples_{self.run_id}.jsonl"
        self.file = open(filepath, "w")
        print(f"[LogExamples] Logger initialized. Writing to {filepath}")

    def log_examples(self, inputs, cont, answers, texts, image_inputs, video_inputs, task, split, doc_id, image_features=None, model=None, processor=None):
        """
        Logs the data for a single batch to the .jsonl file.
        """

        # --- 1. Get Prompt Hidden States from the model output container ---
        prompt_hidden_states = cont.hidden_states[0]

        batch_size = len(answers)

        # --- 2. Loop Over Each Sample in the Batch ---
        for i in range(batch_size):
            #try:
                # --- 2a. Extract Data for Sample 'i' ---
                answer = answers[i]
                input_ids_sample = inputs['input_ids'][i] # Shape: (seq_len)
                hidden_states_sample = prompt_hidden_states[i] # Shape: (seq_len, hidden_dim)

                question = texts[i]
                sample_task = task[i]
                sample_split = split[i]
                sample_doc_id = doc_id[i]

                # --- 2b. Prepare Image for JSON (as Base64 string) ---
                image_b64 = []
                if image_inputs:
                    for image in image_inputs:
                        try:
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format='PNG') # Save as PNG bytes
                            image_bytes = img_buffer.getvalue()
                            im_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        except Exception as e:
                            print(f"Warning: Could not convert image to bytes for sample {i} "
                                f"(global index {self.global_counter}): {e}")
                        image_b64.append(im_b64)

                # --- 2c. Prepare Image Features for JSON (as List) ---
                # split into 3 chunks
                image_features_list = []
                if image_features is not None:
                    image_features_list = image_features.split(image_features.shape[0]//3)
                    image_features_list = [im_feats.mean(axis=0).float().cpu().detach().numpy().tolist() for im_feats in image_features_list]

                # --- 2d. Extract Latent States (as list) ---
                latent_states_list = None
                latent_indices = (input_ids_sample == self.latent_token_id).nonzero(as_tuple=True)[0]
                # pdb.set_trace()
                if latent_indices.numel() > 0:
                    latent_states_tensor = hidden_states_sample.index_select(0, latent_indices)
                    with torch.no_grad():
                        # check the lm_head outputs
                        lm_head_outputs = model.lm_head(latent_states_tensor)
                        # get the top 3 probs from the lm_head outputs
                        top_3_probs = lm_head_outputs.topk(5)
                        # print(top_3_probs)
                        # decode to text
                        decoded_text = processor.batch_decode(top_3_probs.indices)
                    # Convert to list for JSON serialization
                    latent_states_list = latent_states_tensor.float().cpu().detach().numpy().tolist() 
                else:
                    print(f"Warning: No latent tokens found for sample {i} (global index {self.global_counter}).")

                # --- 2e. Create Data Dictionary ---
                data_entry = {
                    "global_sample_index": self.global_counter,
                    "batch_index": i,
                    "task": sample_task,
                    "split": sample_split,
                    "doc_id": sample_doc_id,
                    "question": question,
                    "answer": answer,
                    "image_b64": image_b64, 
                    "latent_states": latent_states_list,
                    "image_features_list": image_features_list, #[1:],
                    # "decoded_text": decoded_text,
                }

                # --- 2f. Write to file and flush ---
                self.file.write(json.dumps(data_entry) + "\n")
                self.file.flush()

            #except Exception as e:
            #    print(f"CRITICAL: Failed to process and log sample {i} "
            #          f"(global index {self.global_counter}): {e}")

            # Increment global counter *after* processing each sample
                self.global_counter += 1



class Qwen2_5_VL_MMLatent_simple(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        sys.path.append("../models")
        Qwen2_5_VLForConditionalGeneration = importlib.import_module("mmlatentdiscrete_qwen_vl").Qwen2_5_VLForConditionalGeneration
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)

        # Add tokens and track if any new tokens were added
        num_added = 0
        num_added += self.processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        num_added += self.processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        num_added += self.processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        num_added += self.processor.tokenizer.add_tokens("<|latent_image|>", special_tokens=True)

        latent_token_idx = self.processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = self.processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = self.processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        imagelatent_idx = self.processor.tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
        self._model.config.latent_token_id = int(latent_token_idx)
        self._model.config.latent_start_id = int(latent_start_idx)
        self._model.config.latent_end_id = int(latent_end_idx)
        self._model.config.imagelatent_token_id = int(imagelatent_idx)

        # Only resize embeddings if new tokens were added
        if num_added > 0:
            self._model.resize_token_embeddings(len(self.processor.tokenizer))
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        
        # current date time as run id. very hacky for now, fix later.
        if self.accelerator.is_local_main_process:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_latents_examples = LogExamples(run_id="MMLatentDiscrete"+run_id, latent_token_id=self._model.config.latent_token_id)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


@register_model("qwen2_5_vl_mmlatent")
class Qwen2_5_VL_MMLatentDiscrete(Qwen2_5_VL_MMLatent_simple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]

            # Add latent padding to the messages
            new_batched_messages = []
            for chat_message in batched_messages:
                if gen_kwargs.get("prompt_version", "old") == "old":
                    if gen_kwargs.get("prompt_mode", "") == "mmlatent2":
                        chat_message.append(
                            {
                                "role": "assistant",
                                "content":
                                [{
                                    "type": "text",
                                    "text": "<think>" + "<|latent_pad|>"*gen_kwargs.get("num_latents", 20) + "</think>"
                                }]
                            }
                        )
                    chat_message.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": ""}]
                        })
                else:
                    if gen_kwargs.get("prompt_mode", "") == "mmlatent2":
                        chat_message.append(
                            {
                                "role": "assistant",
                                "content":
                                [{
                                    "type": "text",
                                    "text": "<think>" + "<|latent_pad|>"*gen_kwargs.get("num_latents", 20) + "</think>\n"
                                }]
                            }
                        )
                    elif gen_kwargs.get("prompt_mode", "") == "new_sepprompt":
                        chat_message.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": " Think using both text and by imagining images to come up with the answer. <think>"}]
                            })
                    elif gen_kwargs.get("prompt_mode", "") == "new_wtxt":
                        chat_message.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": "<think>" + "<|latent_pad|>"*gen_kwargs.get("num_latents", 20)}]
                            })
                    else:
                        chat_message.append(
                            {
                                "role": "assistant",
                                "content":
                                [{
                                    "type": "text",
                                    "text": ""
                                }]
                            }
                        )
                new_batched_messages.append(chat_message)
            batched_messages = new_batched_messages
            # pdb.set_trace()
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in batched_messages]
            texts = [text.replace("<|im_end|>\n", "") for text in texts]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": gen_kwargs.get("max_new_tokens", 128),
                "temperature": 0.01,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            # pdb.set_trace()
            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
                output_hidden_states=current_gen_kwargs.get("output_hidden_states", False),
                return_dict_in_generate=current_gen_kwargs.get("output_hidden_states", False),
            )
            end_time = time.time()
            if current_gen_kwargs.get("output_hidden_states", False):
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont["sequences"])]
            else: 
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            ### log all the raw inputs and outputs for analysis if output hidden states###
            if current_gen_kwargs.get("output_hidden_states", False):
                # this will only work with the BLINK Jigsaw task  - just for analysis, commment out normally
                image_feats = None
                # with torch.no_grad():
                #    image_feats = self.model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
                # pdb.set_trace()
                self.log_latents_examples.log_examples(  # <-- Renamed to 'log_examples'
                    inputs=inputs,
                    cont=cont,         # <-- Pass 'cont' object here
                    answers=answers,
                    texts=texts,
                    image_inputs=image_inputs,
                    video_inputs=video_inputs,
                    task=task,
                    split=split,
                    doc_id=doc_id,
                    image_features=image_feats,
                    model = self.model,
                    processor = self.processor,
                )
            
            # pdb.set_trace()
            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
