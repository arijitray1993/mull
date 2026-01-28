# pylint: disable=all
import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger
import re
import pdb

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # pdb.set_trace()
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")

    video_path = [video_path]
    return video_path


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt_mode = lmms_eval_specific_kwargs["prompt_mode"]

    QUESTION_TEMPLATE_R1 = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
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

    QUESTION_TEMPLATE_BASE = (
        "{Question}\n"
    )

    QUESTION_TEMPLATE_BASE_REASON = (
        "{Question}\n"
        "Please think about this question deeply. "
        "It's encouraged to include self-reflection or verification in the reasoning process."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    }

    TYPE_TEMPLATE_base = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) as the final answer.", # within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) as the final answer.", # within the <answer> </answer> tags.",
    }

    q_type = "free-form"
    if doc["question_type"] in NA_QUESTION_TYPES:
        q_type = "numerical"

    if doc["question_type"] in MCA_QUESTION_TYPES:
        q_type = "multiple choice"

    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(Question=doc["question"]) + TYPE_TEMPLATE[q_type]
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(Question=doc["question"]) + TYPE_TEMPLATE[q_type]
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(Question=doc["question"]) + TYPE_TEMPLATE[q_type]
    elif prompt_mode == "base":
        prompt = QUESTION_TEMPLATE_BASE.format(Question=doc["question"]) + TYPE_TEMPLATE_base[q_type]
    elif prompt_mode == "base_reason":
        prompt = QUESTION_TEMPLATE_BASE_REASON.format(Question=doc["question"]) + TYPE_TEMPLATE_base[q_type]
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return prompt


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    #if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
    #    eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
    return dataset.shuffle(seed=42)
    #return dataset


def fuzzy_matching(pred, gt, doc):
    cleaned_pred = pred.split(" ")[0].rstrip(".").strip()
    patterns = [
        # Pattern for phrases like "the answer is...", "final answer:", etc.
        r"(?:the answer is|the correct answer is|the count is|final answer:)\s+([a-zA-Z0-9]+)",
        # Pattern for phrases like "I counted a total of..."
        r"I counted a total of\s+(\d+)",
        r"count is\s+(\d+)",
        # --- NEW LINE: Pattern for answers inside <answer> tags ---
        r"<answer>\s*(.*?)\s*</answer>"
    ]

    ans_word = cleaned_pred
    # Search in full prediction text, not the truncated cleaned_pred
    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            # The match.group(1) returns the content of the first parenthesis
            ans_word = match.group(1).strip()

    if ans_word.lower() not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
        # extract a number from the answer
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ans_word)
        if numbers:
            ans_word = numbers[-1]
            return ans_word
        else:
            #extract the last capital option letter in pred
            matches = re.findall(r'\b[A-Z]\b', pred)
            if matches:
                pred_letter = matches[-1]
                return pred_letter
            return ans_word
    else:
        return ans_word

def extract_answer(text, gt, doc):
    # return fuzzy_matching(text, gt, doc)

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return fuzzy_matching(text, gt, doc)

def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def vsibench_process_results(doc, results):
    doc["prediction"] = results[0]
    # pdb.set_trace()
    print(doc["prediction"], doc["ground_truth"])
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(extract_answer(doc["prediction"], doc["ground_truth"], doc), doc["ground_truth"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(extract_answer(doc["prediction"], doc["ground_truth"], doc)), to_float(doc["ground_truth"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")
    return {"vsibench_score": doc}


def vsibench_aggregate_results(results):
    # pdb.set_trace()
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == "success_rate":
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    output["object_rel_direction_accuracy"] = (
        sum(
            [
                output.pop("object_rel_direction_easy_accuracy"),
                output.pop("object_rel_direction_medium_accuracy"),
                output.pop("object_rel_direction_hard_accuracy"),
            ]
        )
        / 3.0
    )

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output
