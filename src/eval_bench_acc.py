# pylint: disable=all
"""Eval script for Video-R1."""
from collections import defaultdict
import argparse
import importlib
import json
import os
import pdb
import re
import sys
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import wandb
from torch.utils.data import DataLoader


def extract_think(output_str):
  pattern = r"<think>\s*(.*?)\s*</think>"
  match = re.search(pattern, output_str, re.DOTALL)
  if match:
    return match.group(1).strip()
  return ""

def extract_answer(text):
  pattern = r"<answer>\s*(.*?)\s*</answer>"
  match = re.search(pattern, text, re.DOTALL)
  if match:
    return match.group(1).strip()
  return ""

def normalize_number(num_str):
  try:
    num_str = num_str.replace(",", "")
    return float(num_str)
  except Exception as e:
    return None

def mean_relative_accuracy(
    pred, target, start=0.5, end=0.95, interval=0.05
):

  if not torch.is_tensor(pred):
    pred = torch.tensor(pred, dtype=torch.float32)
  if not torch.is_tensor(target):
    target = torch.tensor(target, dtype=torch.float32)

  epsilon = 1e-8
  rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

  thresholds = torch.arange(
      start, end + interval / 2, interval, dtype=torch.float32
  )

  conditions = rel_error < (1 - thresholds)
  mra = conditions.float().mean()
  return mra.item()

def reward_fn(sample, model_output, question_type):
  try:
    output_ans = extract_answer(model_output)
    if output_ans == "":
      output_ans = model_output
    gt_ans = extract_answer(sample.get("solution", ""))
    if question_type == "multiple choice":
      return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
    elif question_type == "numerical":
      gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
      out_has_decimal = ("." in output_ans) or ("," in output_ans)
      if gt_has_decimal != out_has_decimal:
        return 0.0
      gt_number = normalize_number(gt_ans)
      out_number = normalize_number(output_ans)
      if gt_number is None or out_number is None:
        return 0.0
      return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
    elif question_type == "regression":
      gt_number = normalize_number(gt_ans)
      out_number = normalize_number(output_ans)
      if gt_number is None or out_number is None:
        return 0.0
      mra = mean_relative_accuracy(out_number, gt_number)
      return mra
    else:
      return 0.0
  except Exception as e:
    return 0.0


def main():
  BSZ = 1  # batch size, only tested with 1 use higher with risk. 

  parser = argparse.ArgumentParser(description="Evaluation benchmark")
  parser.add_argument(
      "--eval_conf", type=str, required=True, help="Path to the eval config file"
  )
  args = parser.parse_args()

  exp_confs = json.load(open(args.eval_conf))

  all_dataset_entries = [
      {
          "dataset_name": "vsibench",
          "vsibench_location": (
              "/home/jupyter/data_files/Video-R1-eval/eval_vsibench.json"
          ),
          "vsibench_root": "/home/jupyter/data_files/VSI-Bench", # location of videos
      },
      {
          "dataset_name": "blink",
          "blink_location": (
              "/home/jupyter/data_files/BLINK"
          ),
      },
      {
          "dataset_name": "sat",
          "sat_location": (
              "/home/jupyter/data_files/SAT_new"
          ),
      },
      {
          "dataset_name": "mmvu",
          "mmvu_location": (
              "/home/jupyter/data_files/Video-R1-eval/eval_mmvu.json"
          ),
          "mmvu_root": "/home/jupyter/data_files/MMVU", # location of videos
      },
  ]

  file_name = exp_confs.file_name

  eval_run = wandb.init(
      project=file_name,
      mode="offline"
  )

  ## Model initialization
  sys.path.append("models/")
  get_model = importlib.import_module(
      "get_model"
  ).get_model

  ## DATA INITIALIZATION
  sys.path.append("dataloaders/")
  print(args.dataset_names)
  dataset_entries = [
      entry for entry in all_dataset_entries if entry["dataset_name"] in args.dataset_names
  ]
  data_formatter = importlib.import_module(
      "custom_datasets"
  ).DataFormatter({}, processor, mode="test")
  prepare_dataset = data_formatter.prepare_dataset
  collate_fn = data_formatter.collate_fn

  #sampling_params = SamplingParams(
  #    temperature=0.1,
  #    top_p=0.001,
  #    max_tokens=1024,
  #    stop_token_ids=[],
  #)

  # prepare accelerator for distributed inference
  timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
  accelerator = Accelerator(
      kwargs_handlers=[
          DistributedDataParallelKwargs(find_unused_parameters=True),
          timeout,
      ],
  )

  for model_entry in exp_confs.models:
    model_path = model_entry["model_path"]
    model_name = model_entry["model_name"]
    model_config = model_entry["model_config"]

    model_args = {"model_name": model_name, "model_path": model_path}
    model = get_model(model_args, model_config)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    file_name = model_name
    eval_run.log(
        {
            "model_name": model_name,
            "model_path": model_path,
        }
    )

    for dataset_entry in dataset_entries:
      dataset_name = dataset_entry["dataset_name"]

      eval_run.log(
          {
              "dataset_name": dataset_name,
          }
      )

      OUTPUT_PATH = f"./src/r1-v/eval_results/eval_{dataset_name}_{file_name}_greedy_output.json"

      EvalDataset = importlib.import_module("custom_datasets").EvalDataset
      data = EvalDataset(dataset_entry)
      formatted_data = []
      for entry in data:
        formatted_data.append(prepare_dataset(entry))

      dataloader = DataLoader(data, batch_size=BSZ, collate_fn=collate_fn)

      dataloader, model = (
          accelerator.prepare(
              dataloader, model
          )
      )
      model.eval()

      final_output = []

      mean_acc = []
      mean_mra = []
      mean_acc_bytype = defaultdict(list)
      mean_mra_bytype = defaultdict(list)
      for batch in dataloader:
        inputs = batch
        generated_ids = model.generate(**inputs, temperature=0.1, max_length=1024, top_p=0.001)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        generated_texts = self.accelerator.gather_for_metrics([
            generated_text,
        ])
        batches = self.accelerator.gather_for_metrics([
            batch,
        ])

        for j, (sample, model_output) in enumerate(zip(batches, generated_texts)):
          think_chain = extract_think(model_output)
          final_ans = extract_answer(model_output)
          if final_ans == "":
            final_ans = model_output

          question = sample.get("prompt")
          sample["output"] = model_output
          sample["prediction"] = final_ans
          q_type = sample.get("problem_type", "")
          sample["reward"] = reward_fn(sample, model_output, q_type)
          sample["correct"] = True if sample["reward"] == 1.0 else False
          qa_type = sample.get("original_question_type", "")

          if accelerator.is_main_process:
            print(think_chain, final_ans, sample["reward"])

          if sample["problem_type"] != "regression":
            mean_acc.append(sample["reward"])
            mean_acc_bytype[qa_type].append(sample["reward"])
          else:
            mean_mra.append(sample["reward"])
            mean_mra_bytype[qa_type].append(sample["reward"])
          if think_chain:
            sample["process"] = f"<think>{think_chain}</think>"
          final_output.append(sample)
          # pdb.set_trace()

      try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
          json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
      except Exception as e:
        print(f"Error writing to output file: {e}")

      final_acc = {"mean_acc": 0.0, "mean_mra": 0.0}
      final_acc["mean_acc"] = torch.tensor(mean_acc).mean().item()
      if mean_mra != []:
        final_acc["mean_mra"] = torch.tensor(mean_mra).mean().item()
      print(f"Final accuracy: {final_acc}")

      # by type
      final_acc_bytype = {}
      final_acc_bytype["mean_acc"] = {}
      final_acc_bytype["mean_mra"] = {}
      for qa_type in mean_acc_bytype:
        final_acc_bytype["mean_acc"][qa_type] = torch.tensor(mean_acc_bytype[qa_type]).mean().item()
        if mean_mra_bytype[qa_type] != []:
          final_acc_bytype["mean_mra"][qa_type] = torch.tensor(mean_mra_bytype[qa_type]).mean().item()
      print(f"Final accuracy by type: {final_acc_bytype}")

      # log to wandb
      # mean for the dataset name
      wandb.log(
          {f"{dataset_name}_mean_acc": final_acc["mean_acc"]}
      )
      # mean mra
      if mean_mra != []:
        wandb.log(
            {f"{dataset_name}_mean_mra": final_acc["mean_mra"]}
        )

      # by type
      for qa_type in final_acc_bytype["mean_acc"]:
          wandb.log(
              {f"{dataset_name}_{qa_type}_mean_acc": final_acc_bytype["mean_acc"][qa_type]}
          )
      for qa_type in final_acc_bytype["mean_mra"]:
          wandb.log(
              {f"{dataset_name}_{qa_type}_mean_mra": final_acc_bytype["mean_mra"][qa_type]}
          )


      try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
          json.dump(
              {"results": final_output, "final_acc": [final_acc]},
              f,
              indent=2,
              ensure_ascii=False,
          )
        print(f"Final accuracy saved to {OUTPUT_PATH}")
      except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")

      print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
  main()
