# pylint: disable=all
"""
File for defining your own data class in the Video R1 format
"""
import json
import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import random
from datasets import Dataset as hf_dataset
from datasets import DatasetDict
import pdb
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any
from transformers import Qwen2VLProcessor
from PIL import Image
import re

class DataFormatter:
  def __init__(self, exp_confs, processor, mode="train"):
    self.exp_confs = exp_confs
    self.processor = processor
    self.mode = mode
    self.pause_token = exp_confs.get("pause_token", "<pause>")
    self.query_token = exp_confs.get("query_token", "<query>")

  def prepare_dataset(self, example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
      """Prepare dataset example for training."""

      system_message = "You are a helpful assistant"

      QUESTION_TEMPLATE_R1 = (
          "{Question}\n"
          "Please think about this question as if you were a human pondering deeply. "
          "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
          "It's encouraged to include self-reflection or verification in the reasoning process. "
          "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
      )
      
      QUESTION_TEMPLATE_R1_NoThink = (
          "{Question}\n"
      )

      QUESTION_TEMPLATE_LATENT = (
          "{Question}\n"
          "Please think about this question deeply. "
          "It's encouraged to include self-reflection or verification in the reasoning process. "
          "Provide your final answer between the <answer> </answer> tags."
      )

      QUESTION_TEMPLATE_SFT = (
          "{Question}\n"
          "Provide your final answer between the <answer> </answer> tags."
      )

      TYPE_TEMPLATE = {
          "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
          "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
          "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
          "free-form": " Please provide your text answer within the <answer> </answer> tags.",
          "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
      }

      if example["problem_type"] == 'multiple choice':
        if 'options' in example:
          question = example['problem'] + "Options:\n"
          for op in example["options"]:
              question += op + "\n"
        else:
          question = example['problem']
      else:
          question = example['problem']

      content_multimedia = []
      num_images = 0
      num_videos = 0
      for im_vid_entry in example['multimedia']:
          content_multimedia.append(
              {
                  "type": im_vid_entry['data_type'],
                  im_vid_entry['data_type']: im_vid_entry['path'],
                  # "max_pixels": 128*128,
                  # "fps": 1.0
              }
          )
          if im_vid_entry['data_type'] == 'image':
            num_images += 1
          elif im_vid_entry['data_type'] == 'video':
            num_videos += 1

      image_labels = None
      if 'multimedia_labels' in example:
        if example['multimedia_labels'] is not None:
          if len(example['multimedia_labels'])>0:
            image_labels = example['multimedia_labels']

      if self.exp_confs.get("query_mode", False):
        SUFFIX = "<query>"*(num_images + num_videos*16)
        # content_multimedia += content_multimedia
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content":
                    content_multimedia +
                    [{
                        "type": "text",
                        "text": QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                    },]
            },
        ]
        messages.append({
            "role": "assistant",
            "content":
              [{
                  "type": "text",
                  "text": SUFFIX
              }] + content_multimedia
        })

        if self.mode == "train":
          messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": example['solution']}]
          })
      elif self.exp_confs.get("mmlatent_mode_stage1", False) or self.exp_confs.get("mmlatent_mode_stage2", False) or self.exp_confs.get("mmlatent_mode_stage1_cont", False) or self.exp_confs.get("mmlatent_mode_stage1_imonly", False):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content":
                    content_multimedia +
                    [{
                        "type": "text",
                        "text": QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                    },]
            },
        ]
        if self.mode=="train":
          if self.exp_confs.get("mmlatent_rl_mode", False):
            #example['process'] = example['process'].replace("</think>", "")
            messages.append({
                "role": "assistant",
                "content": 
                  [{
                      "type": "text",
                      "text": example['process'] + "\n"
                  }]
            })
          elif self.exp_confs.get("mmlatentimonly_rl_mode", False):
            messages.append({
                  "role": "assistant",
                  "content": 
                    [{
                        "type": "text",
                        "text": "<think>"
                    }]
            })
          elif self.exp_confs.get("mmlatentimonlyprompt_rl_mode", False):
            messages.append({
                  "role": "assistant",
                  "content": 
                    [{
                        "type": "text",
                        "text": " Think using both text and by imagining images to come up with the answer. <think>"
                    }]
            })
          else:  
            messages.append({
                  "role": "assistant",
                  "content": 
                    [{
                        "type": "text",
                        "text": example['process'] + "\n" + example['solution']
                    }]
            })
      elif self.exp_confs.get("r1_sepprompt_mode", False):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content":
                    content_multimedia +
                    [{
                        "type": "text",
                        "text": QUESTION_TEMPLATE_R1_NoThink.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                    },]
            },
        ]
        if self.mode=="train":
          if self.exp_confs.get("r1_rl_mode", False):
            messages.append({
                  "role": "assistant",
                  "content": [{"type": "text", "text": ""}]
            })
          else:
            messages.append({
                  "role": "assistant",
                  "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]
            })
      elif self.exp_confs.get("r1_mode", False):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content":
                    content_multimedia +
                    [{
                        "type": "text",
                        "text": QUESTION_TEMPLATE_R1.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                    },]
            },
        ]
        if self.mode=="train":
          if self.exp_confs.get("r1_rl_mode", False):
            messages.append({
                  "role": "assistant",
                  "content": [{"type": "text", "text": ""}]
            })
          else:
            messages.append({
                  "role": "assistant",
                  "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]
            })
      else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content":
                    content_multimedia +
                    [{
                        "type": "text",
                        "text": QUESTION_TEMPLATE_SFT.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                    },]
            },
        ]
        if self.mode=="train":
          messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": example['solution']}]
          })
      return {"messages": messages, "image_labels": image_labels, 'solution': example['solution'], "problem_type": example['problem_type']}

  def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
      """Collate batch of examples for training."""
      texts = []
      all_video_inputs = []
      all_image_inputs = []
      all_image_labels = []
      for i, example in enumerate(examples):
        try:
            text_entry = self.processor.apply_chat_template(example["messages"], tokenize=False)
            texts.append(text_entry)
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            if image_inputs is not None:
              all_image_inputs.append(image_inputs)
            if video_inputs is not None:
              all_video_inputs.append(video_inputs)
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")
        if example["image_labels"] is not None:
          all_image_labels.append(example["image_labels"])

      if len(all_image_inputs) == 0:
        all_image_inputs = None

      if len(all_video_inputs) == 0:
        all_video_inputs = None

      inputs_image_labels = None
      if len(all_image_labels) > 0:
        inputs_image_labels = self.processor.image_processor(images=all_image_labels, return_tensors="pt")

      inputs = self.processor(
          text=texts,
          images=all_image_inputs,
          videos=all_video_inputs,
          return_tensors="pt",
          padding=True
      )
      labels = inputs["input_ids"].clone()
      labels[labels == self.processor.tokenizer.pad_token_id] = -100

      # set the <|latent_pad|>, <|latent_start|> and <|latent_end|>, <|latent_image|> token to -100
      labels[labels == self.processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>")] = -100
      # labels[labels == self.processor.tokenizer.convert_tokens_to_ids("<|latent_start|>")] = -100
      # labels[labels == self.processor.tokenizer.convert_tokens_to_ids("<|latent_end|>")] = -100
      labels[labels == self.processor.tokenizer.convert_tokens_to_ids("<|latent_image|>")] = -100

      # Handle visual tokens based on processor type
      visual_tokens = [151652, 151653, 151656] if isinstance(self.processor, Qwen2VLProcessor) else [
          self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
      ]

      for visual_token_id in visual_tokens:
          labels[labels == visual_token_id] = -100

      inputs["labels"] = labels

      if inputs_image_labels is not None:
        inputs['pixel_values_latent'] = inputs_image_labels['pixel_values']
        inputs['image_grid_thw_latent'] = inputs_image_labels['image_grid_thw']

        # make an image_out_mask same shape as input_ids wherever input_ids is <|latent_image|>
        image_out_mask = inputs['input_ids'] == self.processor.tokenizer.convert_tokens_to_ids("<|latent_image|>")
        inputs['image_out_mask'] = image_out_mask

      # the keys are:
      # 'input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'second_per_grid_ts', 'labels'
      # pdb.set_trace()
      return inputs


class CustomMix(Dataset):
  def __init__(self, args):
    self.args = args

    self.all_mix = []
    self.all_lens = []
    self.weights = []

    mix_datas = args.get("mix_datas")

    if "SAT" in mix_datas:
      self.procthor_data = SAT(args)
      self.all_mix.append(self.procthor_data)
      self.weights.append(mix_datas["SAT"])
      self.all_lens.append(len(self.procthor_data))

    if "ZebraCOT" in mix_datas:
      self.zebracot_data = ZebraCOT(args)
      self.all_mix.append(self.zebracot_data)
      self.weights.append(mix_datas["ZebraCOT"])
      self.all_lens.append(len(self.zebracot_data))

    if "VideoR1" in mix_datas:
      self.video_r1_data = VideoR1(args)
      self.all_mix.append(self.video_r1_data)
      self.weights.append(mix_datas["VideoR1"])
      self.all_lens.append(len(self.video_r1_data))
    
    if "SIMS" in mix_datas:
      self.sims_data = SIMS_v1(args)
      self.all_mix.append(self.sims_data)
      self.weights.append(mix_datas["SIMS"])
      self.all_lens.append(len(self.sims_data))

    print("combined data ...")

    print("Total number of data points: ", sum(self.all_lens))


  def __getitem__(self, idx):
    if not (0 <= idx < sum(self.all_lens)):
      raise IndexError("Index out of bounds")
    data = random.choices(population=self.all_mix, k=1, weights=self.weights)[0]
    entry = data[idx % len(data)]
    return entry

  def __len__(self):
    return sum(self.all_lens)


class SAT(Dataset):
  """
  The __getitem__ function must return a dict with the following keys:
  - problem_type
  - multimedia - list of dicts with images or videos.
  - problem
  - options if problem_type is multiple_choice.
  - process
  - solution
  """

  def __init__(self, args):
    self.args = args
    split = args.get("split", "train")

    # Load from HuggingFace if sat_location starts with a HF repo name, otherwise load local parquet
    sat_location = args.get("sat_location", "array/SAT")
    if "/" in sat_location and not os.path.exists(sat_location):
      # Load from HuggingFace - dataset has images directly included
      dataset = load_dataset(sat_location, trust_remote_code=True)
      self.sat_data = dataset[split]
      self.use_hf_images = True
    else:
      # Load from local parquet
      dataset = load_dataset("parquet", data_files=os.path.join(sat_location, f"SAT_{split}.parquet"))
      self.sat_data = dataset[split]
      self.use_hf_images = False

    self.ind_to_letter = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
    }

    data_formatter = DataFormatter(args, args["processor"], mode=args["mode"])
    self.prepare_dataset = data_formatter.prepare_dataset

  def __getitem__(self, idx):
    entry = self.sat_data[idx]

    multimedia_entry = []
    if self.use_hf_images:
      # Images come directly from HuggingFace dataset as PIL images
      images = entry.get("images", [])
      for image in images:
        multimedia_entry.append({
            "data_type": "image",
            "image": image,  # PIL image directly
        })
    else:
      # Load images from local paths
      images = entry["image_paths"]
      for image in images:
        im_path = os.path.join(self.args["sat_location"], image.split("SAT_new/")[-1])
        multimedia_entry.append({
            "data_type": "image",
            "path": im_path,
        })

    question = entry["question"]

    corrected_answer_choices = []
    answer_choices = entry["answers"]
    for answer in answer_choices:
      if (
          "in the first frame" in answer
      ):  # a small bug, todo fix in data generation later.
          answer = answer.replace("in the first frame", "")
          corrected_answer_choices.append(answer)
      else:
        corrected_answer_choices.append(answer)

    correct_answer = entry["correct_answer"].replace("in the first frame", "")

    answer_choices_shuffled = corrected_answer_choices
    random.shuffle(answer_choices_shuffled)
    answer_choices = [f"{self.ind_to_letter[i]}. {ans}" for i, ans in enumerate(answer_choices_shuffled)]

    # index of the correct answer in the shuffled list
    correct_answer_index = answer_choices_shuffled.index(correct_answer)
    correct_answer_letter = self.ind_to_letter[correct_answer_index] + ". " + correct_answer

    if self.args.get("use_sep_textim_reason_prompts"):
      process = " Directly predict the answer without thinking. "
    elif self.args.get("use_sep_text_reason_prompts"):
      process = " Directly predict the answer without thinking. "
    else:
      process = ""

    if self.args.get("mmlatent_mode_stage2", False):
      process = "<|latent_pad|>"*self.args.get("num_latent_tokens", 20)
      process = "<think>" + process + "</think>"

    qa_type = entry["question_type"]

    data_entry = {
        "problem_type": "multiple choice",
        "problem": question,
        "options": answer_choices,
        "process": process,
        "solution": "<answer>" + correct_answer_letter + "</answer>",
        "multimedia": multimedia_entry,
        "original_question_type": qa_type,
    }

    prepared_entry = self.prepare_dataset(data_entry)
    return prepared_entry

  def __len__(self):
    return len(self.sat_data)


class ZebraCOT(Dataset):
  def __init__(self, args):
    self.args = args
    all_tasks = [
        "2D Visual Reasoning - Visual Jigsaw",
        "2D Visual Reasoning - Visual Search",
        "3D Visual Reasoning - Embodied CoT",
        "3D Visual Reasoning - Multi-Hop Objects Counting",
        "3D Visual Reasoning - Robot Planning",
        "Scientific Reasoning - Chemistry",
        "Scientific Reasoning - Competitive Programming",
        "Scientific Reasoning - Geometry",
        "Scientific Reasoning - Graph Algorithms",
        "Scientific Reasoning - Physics",
        "Visual Logic & Strategic Games - ARC-AGI",
        "Visual Logic & Strategic Games - Checkers",
        "Visual Logic & Strategic Games - Chess",
        "Visual Logic & Strategic Games - Ciphers",
        "Visual Logic & Strategic Games - Connect Four",
        "Visual Logic & Strategic Games - Maze",
        "Visual Logic & Strategic Games - RPM",
        "Visual Logic & Strategic Games - Tetris",
    ]
    all_data = []
    all_lens = []
    for data_split in all_tasks:
      dataset = load_dataset(args.get("zebracot_location"), data_split)
      data_len = len(dataset)
      all_data.append(dataset)
      all_lens.append(data_len)

    self.data = all_data
    self.lens = all_lens

    data_formatter = DataFormatter(args, args["processor"], mode=args["mode"])
    self.prepare_dataset = data_formatter.prepare_dataset

  def __len__(self):
    return sum(self.lens)

  def __getitem__(self, idx):
    if not (0 <= idx < sum(self.lens)):
      raise IndexError("Index out of bounds")
    data_choice = random.choices(population=self.data, k=1, weights=self.lens)[0]['train']
    data_entry = data_choice[idx % len(data_choice)]

    # format the data entry
    format_entry = self.reformat_dictionary(data_entry)
    if format_entry is None:
      print("skipping entry ZebraCOT")
      return self.__getitem__((idx+1)%sum(self.lens))

    # prepare the entry
    prepared_entry = self.prepare_dataset(format_entry)
    return prepared_entry

  def reformat_dictionary(self, data: dict) -> dict:
    """
    Reformats the dictionary to interleave text tokens and group images.
    """

    if self.args.get("mmlatent_mode_stage1", False):
      # 1. Process the 'Text Reasoning Trace'
      trace_text = data.get('Text Reasoning Trace', '')

      # Replace all image placeholders with <|latent_image|>
      trace_with_latent_images = re.sub(
          r'<image_start>\[reasoning_image_\d+\]<image_end>',
          '<|latent_image|>',
          trace_text
      )

      # Split the entire modified string by whitespace and join with the pad token
      words = trace_with_latent_images.split()
      processed_trace = '<|latent_pad|>'.join(words)
      processed_trace = "<think>" + processed_trace + "</think>"
    elif self.args.get("mmlatent_mode_stage1_imonly", False):
      trace_text = data.get('Text Reasoning Trace', '')

      # Replace all image placeholders with <|latent_image|>
      trace_with_latent_images = re.sub(
          r'<image_start>\[reasoning_image_\d+\]<image_end>',
          '<|latent_start|><|latent_image|>',
          trace_text
      )

      # choose a random point to insert the latent pause in the trace_with_latent_images
      #trace_words = trace_with_latent_images.split()
      # choose a random point to insert the latent pause in the trace_with_latent_images
      #random_index = random.randint(0, len(trace_words)-2)
      # processed_trace = " ".join(trace_words[:random_index]) + "<|latent_pad|>" + " ".join(trace_words[random_index:])
      processed_trace = "<think>" + trace_with_latent_images + "</think>"
    elif self.args.get("mmlatent_mode_stage1_cont", False):
      trace_text = data.get('Text Reasoning Trace', '')

      # Replace all image placeholders with <|latent_image|>
      trace_with_latent_images = re.sub(
          r'<image_start>\[reasoning_image_\d+\]<image_end>',
          '<|latent_image|>',
          trace_text
      )

      # choose a random point to insert the latent pause in the trace_with_latent_images
      trace_words = trace_with_latent_images.split()
      # choose a random point to insert the latent pause in the trace_with_latent_images
      random_index = random.randint(0, len(trace_words)-2)
      processed_trace = " ".join(trace_words[:random_index]) + "<|latent_pad|>" + " ".join(trace_words[random_index:])
      processed_trace = "<think>" + processed_trace + "</think>"
      # pdb.set_trace()
    elif self.args.get("mmlatent_mode_stage2", False):
      if self.args.get("use_multimodal_textonly", False):
        trace_text = data.get('Text Reasoning Trace', '')

        # Replace all image placeholders with <|latent_image|>
        trace_with_no_images = re.sub(
            r'<image_start>\[reasoning_image_\d+\]<image_end>',
            '',
            trace_text
        )
        latent_pad = "<|latent_pad|>"*self.args.get("num_latent_tokens", 20)
        processed_trace = "<think>" + latent_pad + trace_with_no_images + "</think>"
      elif self.args.get("wtxt_rl_mode", False):
        processed_trace = "<|latent_pad|>"*self.args.get("num_latent_tokens", 20)
        processed_trace = "<think>" + processed_trace
      else:
        processed_trace = "<|latent_pad|>"*self.args.get("num_latent_tokens", 20)
        processed_trace = "<think>" + processed_trace + "</think>"

    elif self.args.get("no_reason_mode", False):
      processed_trace = ""
    elif self.args.get("no_zebracot_reason_mode", False):
      processed_trace = ""
    else:
      raise ValueError("Unsupported mmlatent mode")

    # process the question
    question = data.get('Question')
    # Remove all image placeholders
    question = re.sub(
        r'<image_start>\[problem_image_\d+\]<image_end>',
        '',
        question
    )
    # 2. Collect and sort images
    problem_images = []
    reasoning_images = []

    # Find and sort keys to ensure the image order is correct
    problem_keys = sorted([k for k in data if k.startswith('problem_image_')])
    reasoning_keys = sorted([k for k in data if k.startswith('reasoning_image_')])

    for key in problem_keys:
        problem_images.append(data[key])

    # problem_images = problem_images[:2]
    # pdb.set_trace()# count number of images
    num_im = 0
    for p_image in problem_images:
      if p_image is not None:
        num_im += 1
    multimedia_entry = []
    for p_image in problem_images:
      if p_image is not None:
        if num_im > 2:
          p_image = p_image.resize((128,128), Image.Resampling.LANCZOS)
        else:
          p_image = p_image.resize((512, 512), Image.Resampling.LANCZOS)
        multimedia_entry.append({
            "data_type": "image",
            "path": p_image,
        })
    print("num images", len(multimedia_entry))
    if len(multimedia_entry) < 1:
      return None

    if len(multimedia_entry) > 5:
      return None

    if self.args.get("mmlatent_mode_stage1", False) or self.args.get("mmlatent_mode_stage1_cont", False) or self.args.get("mmlatent_mode_stage1_imonly", False):
      for key in reasoning_keys:
        if data[key] is not None:
          r_img = data[key].resize((128,128), Image.Resampling.LANCZOS)
          reasoning_images.append(r_img)

      if len(reasoning_images)>5:
        return None

    if self.args.get("use_sep_textim_reason_prompts"):
      processed_trace = "Think using both text and by imagining images to come up with the answer. " + processed_trace
    elif self.args.get("use_sep_text_reason_prompts"):
      processed_trace = " Directly predict the answer without thinking. "

    # 3. Build the new dictionary
    output_dict = {
        'problem_type': 'free-form',
        'problem': question,
        'process': processed_trace,
        'solution': "<answer>" + data.get('Final Answer') + "</answer>",
        'multimedia': multimedia_entry,
        'multimedia_labels': reasoning_images,
    }
    # pdb.set_trace()
    return output_dict



class VideoR1(Dataset):
  def __init__(self, args):
    self.args = args
    split = args.get("split", "train")
    dataset = DatasetDict({"train": hf_dataset.from_json(args["video_r1_location"])})
    self.vid_data = dataset[split]
    self.data_root = "/"+os.path.join(*args["video_r1_location"].split("/")[:-1])

    data_formatter = DataFormatter(args, args["processor"], mode=args["mode"])
    self.prepare_dataset = data_formatter.prepare_dataset

  def __getitem__(self, idx):
    entry = self.vid_data[idx]

    im_vid_file = os.path.join(self.data_root, entry['path'])

    data_type = entry['data_type']
    multimedia_entry = [
      {
        "data_type": data_type,
        "path": im_vid_file
      },
    ]

    entry['multimedia'] = multimedia_entry

    process = entry['process']
    process = process.replace("<think>", "")
    process = process.replace("</think>", "")
    if self.args.get("pause_mode", False):
      # interleave pause inbetween each word of the process string
      process_words = process.split(" ")
      process = "".join([f"<pause>{w}" for w in process_words])

    if self.args.get("mmlatent_mode_stage1", False):
      # interleave pause inbetween each word of the process string
      process_words = process.split(" ")
      process = "".join([f"<|latent_pad|>{w}" for w in process_words])

    if self.args.get("mmlatent_mode_stage1_cont", False):
      process_words = process.split(" ")
      random_index = random.randint(0, len(process_words)-1)
      process = " ".join(process_words[:random_index]) + "<|latent_pad|>" + " ".join(process_words[random_index:])

    if self.args.get("mmlatent_mode_stage2", False):
      process = "<|latent_pad|>"*self.args.get("num_latent_tokens", 20)
      if self.args.get("use_text_reasoning", False):
        process = process + " " + entry['process']

    if self.args.get("use_sep_textim_reason_prompts"):
      entry['process'] = "Think only using text to come up with the answer. " + "<think>" + process + "</think>"
    elif self.args.get("use_sep_text_reason_prompts"):
      process = " Think only using text to come up with the answer. " + "<think>" + process + "</think>"
    else:
      entry['process'] = "<think>" + process + "</think>"

    if self.args.get("no_reason_mode", False):
      entry['process'] = ""

    # solution already has <answer> and </answer> tags

    prepared_entry = self.prepare_dataset(entry)
    return prepared_entry

  def __len__(self):
    return len(self.vid_data)


class SIMS_v1(Dataset):
  def __init__(self, args):
    self.args = args
    self.data_root = args["sims_location"]

    if self.args.get("use_3q_40k", False):
      self.data = load_dataset("json", data_files=os.path.join(self.data_root, "sims_3q/sims_3q_40k.jsonl"))
    elif self.args.get("use_200k", False):
      self.data = load_dataset("json", data_files=os.path.join(self.data_root, "sims_200k.jsonl"))
    else:
      self.data = load_dataset("json", data_files=os.path.join(self.data_root, "llava_vid_sims_combined_20k.jsonl"))

    self.data = self.data["train"]
    data_formatter = DataFormatter(args, args["processor"], mode=args["mode"])
    self.prepare_dataset = data_formatter.prepare_dataset

  def __getitem__(self, idx):
    # Example entry:
    """
    {'id': 'spoc_feb19__7852',
      'conversations': [
          { 
            'from': 'human',
            'value': '<image>These are frames of a video.\nMeasuring from the closest point of each object, which of these objects (arm chair, dining table, doorway, fire extinguisher) is the closest to the toilet?\nIf there are multiple instances of an object category, measure to the closest.\nA. fire extinguisher\nB. dining table\nC. doorway\nD. arm chair\nPlease select the correct option from the previous choices.',
            'task': 'vsi_obj_rel_distance',
            'pwdvalue': None
          },
          {
            'from': 'gpt',
            'value': 'A. fire extinguisher',
            'task': None,
            'pwdvalue': None
          }
        ],
      'type': 'mc',
      'data_source': 'spoc_feb19',
      'video': 'train/000935/raw_navigation_camera__1.mp4',
      'source_file': '/datasets/consolidated/llava/sims_v1/qas/train/rgb/mt1_vsi_obj_rel_distance_mc_ablation_seed_0.jsonl',
      'source': 'sims',
      'question_type': 'obj_rel_distance'
    }
    """
    entry = self.data[idx]

    entry['multimedia'] = [{
        "data_type": "video",
        "path": os.path.join(self.data_root, entry['video'])
    }]

    entry['problem'] = entry['conversations'][0]['value']
    if entry['problem'] is None:
      entry['problem'] = ""
      print("WARNING: Question is None")
    entry['problem'] = entry['problem'].replace("<image>", "")
    entry['solution'] = "<answer>" + entry['conversations'][1]['value'].split(".")[0] + "</answer>"

    if self.args.get("mmlatent_mode_stage2", False):
      process = "<|latent_pad|>"*self.args.get("num_latent_tokens", 20)
    else:
      process = ""
    entry['process'] = process

    if entry['type'] == "mc":
      entry['problem_type'] = "multiple choice"
    elif entry['type'] == "oe":
      entry['problem_type'] = "numerical"
    else:
      entry['problem_type'] = "free-form"

    processed_entry = self.prepare_dataset(entry)
    return processed_entry

  def __len__(self):
    return len(self.data)


### Deprecated - evals run through lmms-eval ###
class EvalDataset(Dataset):
  def __init__(self, args):
    self.args = args
    dataset_name = args["dataset_name"]
    if dataset_name == "vsibench":
      self.eval_data = VSIBench(args)
    elif dataset_name == "blink":
      self.eval_data = BLINK(args)
    elif dataset_name == "sat":
      self.eval_data = SAT_test(args)
    elif dataset_name == "mmvu":
      self.eval_data = MMVU(args)
    else:
      raise ValueError(f"Unknown dataset name: {dataset_name}")

  def __getitem__(self, idx):
    return self.eval_data[idx]

  def __len__(self):
    return len(self.eval_data)


class SAT_test(Dataset):
  def __init__(self, args):
    self.args = args
    split = args.get("split", "test")
    dataset = load_dataset("parquet", data_files=os.path.join(args["sat_location"], f"SAT_{split}.parquet"))

    self.sat_data = dataset["train"] # hf loads any specific parquet as train but it indeed is test  # [args['split']]

    # pdb.set_trace()
    self.data = list(self.sat_data)

    # we should do cicular eval. So, we need to copy the data with the answer choices reversed.
    reversed_data = []
    for entry in self.sat_data:
      entry = entry.copy()
      answer_choices = entry["answers"]
      new_answer_choices = answer_choices[::-1]
      entry["answers"] = new_answer_choices
      reversed_data.append(entry)

    self.data += reversed_data

    self.ind_to_letter = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
    }

  def __getitem__(self, idx):
    entry = self.sat_data[idx]

    images = entry["image_paths"]
    # this is a list of images. Some questions are on one image,
    # and some on 2 images

    multimedia_entry = []
    for image in images:
      im_path = os.path.join(self.args["sat_location"], image.split("SAT_new/")[-1])
      multimedia_entry.append({
          "data_type": "image",
          "path": im_path,
      })

    question = entry["question"]

    corrected_answer_choices = []
    answer_choices = entry["answers"]
    for answer in answer_choices:
      if (
          "in the first frame" in answer
      ):  # a small bug, todo fix in data generation later.
          answer = answer.replace("in the first frame", "")
          corrected_answer_choices.append(answer)
    answer_choices = corrected_answer_choices

    answer_choices_shuffled = corrected_answer_choices
    answer_choices = [f"{self.ind_to_letter[i]}. {ans}" for i, ans in enumerate(answer_choices_shuffled)]

    # index of the correct answer in the shuffled list
    correct_answer = answer_choices_shuffled.index(entry["correct_answer"])
    correct_answer_letter = self.ind_to_letter[correct_answer]

    qa_type = entry["question_type"]

    return {
        "problem_type": "multiple choice",
        "problem": question,
        "options": answer_choices,
        "process": "",
        "solution": "<answer>" + correct_answer_letter + "</answer>",
        "multimedia": multimedia_entry,
        "original_question_type": qa_type,
    }

  def __len__(self):
    return len(self.sat_data)


class MMVU(Dataset):
  def __init__(self, args):
    self.args = args
    with open(args["mmvu_location"], "r", encoding="utf-8") as f:
      self.data = json.load(f)

    self.video_root = args["mmvu_root"]

  def __getitem__(self, idx):
    entry = self.data[idx]
    entry["multimedia"] = [{
        "data_type": entry["data_type"],
        "path": self.video_root + "/" + entry["path"].split("/MMVU")[-1],
    }]
    return entry

  def __len__(self):
    return len(self.data)


class VSIBench(Dataset):
  def __init__(self, args):
    self.args = args
    with open(args["vsibench_location"], "r", encoding="utf-8") as f:
      self.data = json.load(f)

    self.video_root = args["vsibench_root"]

  def __getitem__(self, idx):
    entry = self.data[idx]
    #pdb.set_trace()
    # check if entry is list
    if isinstance(entry, list):
      for en in entry:
        en["multimedia"] = [{
            "data_type": en["data_type"],
            "path": self.video_root + "/" + en["path"].split("/VSIBench")[-1],
        }]
    else:
      entry["multimedia"] = [{
          "data_type": entry["data_type"],
          "path": self.video_root + "/" + entry["path"].split("/VSIBench")[-1],
      }]

    return entry

  def __len__(self):
    return len(self.data)

class BLINK(Dataset):
  def __init__(self, args):
    self.args = args

    dataset_name = args.get("blink_location", "BLINK-Benchmark/BLINK")

    SUBTASK_NAME = [
        "Multi-view_Reasoning",
        "Relative_Depth",
        "Spatial_Relation",
    ]  # , 'Object_Localization',]
    # SUBTASK_NAME = ['Relative_Depth', 'Spatial_Relation'] # , 'Object_Localization',]

    self.data = []
    for subtask in SUBTASK_NAME:
      count = 0
      for entry in load_dataset(dataset_name, subtask)["val"]:
        self.data.append((entry, subtask))
        count += 1

    self.choice_to_number = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
    }

  def __getitem__(self, idx):
    entry, subtask = self.data[idx]
    question = entry["prompt"].split("?")[0] + "?"

    # question = question.replace("The images are frames from a video. The video is shooting a static scene. The camera is either moving clockwise (left) or counter-clockwise (right) around the object.", "")

    answer = entry["answer"].replace("(", "").replace(")", "")

    answer_choices = entry["choices"]

    images = []
    image_1 = entry["image_1"]
    images.append(image_1)
    if entry["image_2"] is not None:
      image_2 = entry["image_2"]
      images.append(image_2)

    multimedia_entry = []
    for image in images:
      multimedia_entry.append({
          "data_type": "image",
          "path": image,
      })

    # pdb.set_trace()
    return {
        "problem_type": "multiple choice",
        "problem": question,
        "options": answer_choices,
        "process": "",
        "solution": "<answer>" + answer+ "</answer>",
        "multimedia": multimedia_entry,
        "original_question_type": subtask,
    }

  def __len__(self):
    return len(self.data)
