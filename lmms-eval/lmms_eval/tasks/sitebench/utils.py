# pylint: disable=all
# Adapted from official lmms-eval sitebench implementation

import os
import random
import re
import string
from collections import defaultdict

import datasets
import numpy as np
from PIL import Image

# Maximum number of images allowed per example (to avoid OOM)
MAX_IMAGES_PER_EXAMPLE = 4

UpperLetters = list(string.ascii_uppercase)
Categories = {
    "counting & existence",
    "spatial relationship reasoning",
    "object localization & positioning",
    "depth & 3d understanding",
    "movement navigation & intent prediction",
    "multi-view & cross-image reasoning",
}

# Get the cache directory from HF_HOME
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
cache_dir = os.path.join(base_cache_dir, "sitebench")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter out examples with too many images to avoid OOM errors."""
    original_len = len(dataset)
    dataset = dataset.filter(
        lambda x: len(x["visual"]) <= MAX_IMAGES_PER_EXAMPLE,
        desc="Filtering examples with too many images",
    )
    filtered_len = len(dataset)
    if original_len != filtered_len:
        print(
            f"Filtered {original_len - filtered_len} examples with more than "
            f"{MAX_IMAGES_PER_EXAMPLE} images ({filtered_len}/{original_len} remaining)"
        )
    return dataset


QUESTION_TEMPLATE_R1 = (
    "{Question}\n"
    "Options:\n{Options}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', "
    "'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural "
    "language thought expressions. "
    "It's encouraged to include self-reflection or verification in the reasoning "
    "process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then "
    "give your final answer between the <answer> </answer> tags."
)

QUESTION_TEMPLATE_LATENT = (
    "{Question}\n"
    "Options:\n{Options}\n"
    "Please think about this question deeply. "
    "It's encouraged to include self-reflection or verification in the reasoning "
    "process. "
    "Provide your final answer between the <answer> </answer> tags."
)

QUESTION_TEMPLATE_SFT = (
    "{Question}\n"
    "Options:\n{Options}\n"
    "Provide your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = (
    " Please provide only the single option letter (e.g., A, B, C, D, etc.) "
    "within the <answer> </answer> tags."
)


def sitebench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt_mode = lmms_eval_specific_kwargs.get("prompt_mode", "videor1")
    question = doc["question"].strip()
    options = doc["options"]
    option_text = "\n".join(
        f"{UpperLetters[i]}: {options[i]}" for i in range(len(options))
    )

    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(
            Question=question, Options=option_text
        ) + TYPE_TEMPLATE
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(
            Question=question, Options=option_text
        ) + TYPE_TEMPLATE
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(
            Question=question, Options=option_text
        ) + TYPE_TEMPLATE
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    return prompt


def sitebench_doc_to_visual(doc):
    imgs = []
    for image_path in doc["visual"]:
        full_image_path = os.path.join(cache_dir, image_path)
        imgs.append(Image.open(full_image_path).convert("RGB"))
    return imgs


def extract_answer_letter(pred, all_choices):
    """Extract the answer letter from model prediction."""
    # First try to extract from <answer> tags
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(answer_pattern, pred, re.IGNORECASE | re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        # Extract single letter from answer content
        for choice in all_choices:
            if choice in answer_content.upper():
                return choice
        # If the content is just a letter
        if len(answer_content) == 1 and answer_content.upper() in all_choices:
            return answer_content.upper()

    # Fallback to original parsing logic
    response = " " + pred + " "

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # Look for choices with colons, e.g., A:, B:, C:
    if len(candidates) == 0:
        for choice in all_choices:
            if (
                f"{choice}:" in response
                or f":{choice}" in response
                or f": {choice}" in response
            ):
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def sitebench_process_results(doc, results):
    response = results[0].strip()
    all_choices = UpperLetters[: len(doc["options"])]
    pred_index = extract_answer_letter(response, all_choices)
    gt_index = doc["answer"]
    score = 1.0 if pred_index == gt_index else 0.0

    category = doc["category"]
    dataset = doc["dataset"]
    accuracy_dict = {"overall": score, category: score, dataset: score, "total": 1}

    return {"accuracy": accuracy_dict}


def sitebench_aggregate_results(results):
    total_correct, total_examples = 0, 0
    category_correct, category_total = defaultdict(int), defaultdict(int)
    dataset_correct, dataset_total = defaultdict(int), defaultdict(int)

    for result in results:
        # Overall accuracy
        total_correct += result["overall"]
        total_examples += result["total"]

        # Category accuracy / Dataset accuracy
        for key, score in result.items():
            if key in Categories:
                category_correct[key] += score
                category_total[key] += result["total"]
            elif key not in ("overall", "total"):
                dataset_correct[key] += score
                dataset_total[key] += result["total"]

    overall_accuracy = (
        (total_correct / total_examples) * 100 if total_examples > 0 else 0.0
    )
    category_accuracy = {
        category: (category_correct[category] / category_total[category]) * 100
        if category_total[category] > 0
        else 0.0
        for category in category_correct
    }
    dataset_accuracy = {
        dataset: (dataset_correct[dataset] / dataset_total[dataset]) * 100
        if dataset_total[dataset] > 0
        else 0.0
        for dataset in dataset_correct
    }

    return round(overall_accuracy, 5)
