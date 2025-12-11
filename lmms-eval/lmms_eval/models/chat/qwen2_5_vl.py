# pylint: disable=all
import time
from typing import List, Optional, Tuple, Union
import pdb
import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

import json
import base64
from io import BytesIO
import io
import datetime

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")



class LogExamples:
    def __init__(self, run_id: str):
        """
        Initializes the logger.

        Args:
            run_id (str): A unique ID for this evaluation run.
            latent_token_id (int): The token ID for your latent_pad tokens.
        """
        self.run_id = run_id
        self.global_counter = 0

        # Open a file to store the examples. 
        # Using .jsonl (JSON Lines) as the extension is standard for this format.
        filepath = f"/home/jupyter/vis/examples_{self.run_id}.jsonl"
        self.file = open(filepath, "w")
        print(f"[LogExamples] Logger initialized. Writing to {filepath}")

    def log_examples(self, 
                     answers: List[str], 
                     texts: List[str], 
                     images_batch: List[List[Image.Image]], 
                     task: List[str], 
                     split: List[str], 
                     doc_id: List[str]):
        """
        Logs the simple data (question, answer, images) for a single batch.

        Args:
            answers (List[str]): List of generated answers.
            texts (List[str]): List of input prompts/questions.
            images_batch (List[List[PIL.Image]]): A list where each element
                is a list of PIL Images for the corresponding sample.
            task (List[str]): List of tasks.
            split (List[str]): List of splits.
            doc_id (List[str]): List of document IDs.
        """

        batch_size = len(answers)

        # --- Loop Over Each Sample in the Batch ---
        for i in range(batch_size):
            try:
                # --- 1. Extract Per-Sample Data ---
                answer = answers[i]
                question = texts[i]
                sample_task = task[i]
                sample_split = split[i]
                sample_doc_id = doc_id[i]
                images_for_sample = images_batch # Get the list of images for *this* sample

                # --- 2. Prepare Images for JSON (as Base64 strings) ---
                image_b64_list = []
                if images_for_sample:
                    for image in images_for_sample:
                        try:
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format='PNG') # Save as PNG bytes
                            image_bytes = img_buffer.getvalue()
                            im_b64 = base64.b64encode(image_bytes).decode('utf-8')
                            image_b64_list.append(im_b64)
                        except Exception as e:
                            print(f"Warning: Could not convert image to bytes for sample {i} "
                                  f"(global index {self.global_counter}): {e}")

                # --- 3. Create Data Dictionary ---
                data_entry = {
                    "global_sample_index": self.global_counter,
                    "batch_index": i,
                    "task": sample_task,
                    "split": sample_split,
                    "doc_id": sample_doc_id,
                    "question": question,
                    "answer": answer,
                    "images_b64": image_b64_list,
                }

                # --- 4. Write to file and flush ---
                self.file.write(json.dumps(data_entry) + "\n")
                self.file.flush()

            except Exception as e:
                print(f"CRITICAL: Failed to process and log sample {i} "
                      f"(global index {self.global_counter}): {e}")

            # Increment global counter *after* processing each sample
            self.global_counter += 1



@register_model("qwen2_5_vl_chat")
class Qwen2_5_VL(Qwen2_5_VLSimple):
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
            #new_batched_messages = []
            #for chat_message in batched_messages:
            #    chat_message.append(
            #        {
            #            "role": "assistant",
            #            "content":
            #            [{
            #                "type": "text",
            #                "text": ""
            #            }]
            #        }
            #    )
            #    new_batched_messages.append(chat_message)
            # batched_messages = new_batched_messages
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            texts = [text.replace("<|im_end|>", "") for text in texts]
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
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
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
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            ########## log examples comment out normally
            # if simple logger is not initalized, then initialize it
            if not hasattr(self, 'simple_logger') and current_gen_kwargs.get("log_samples", False):
                run_id = "base_reasonmodel_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.simple_logger = LogExamples(run_id=run_id)
            
            if current_gen_kwargs.get("log_samples", False):
                self.simple_logger.log_examples(
                    answers=answers,
                    texts=texts,
                    images_batch=image_inputs,
                    task=task,
                    split=split,
                    doc_id=doc_id
                )
            ######################################################################
            
            
            
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
