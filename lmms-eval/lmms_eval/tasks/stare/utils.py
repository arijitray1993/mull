# pylint: disable=all
from collections import defaultdict
import re
import pdb
import os
import random
import pandas as pd
from PIL import Image
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

from datasets import Dataset, concatenate_datasets


def stare_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    full_prompt = doc["question"]
    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(Question=full_prompt) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(Question=full_prompt) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(Question=full_prompt) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "base":
        prompt = QUESTION_TEMPLATE_BASE.format(Question=full_prompt)
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    # pdb.set_trace()
    return prompt

def stare_doc_to_visual(doc):
    image_list = doc["images"]
    return image_list


def stare_doc_to_target(doc):
    gt_answer = doc["answer"]
    return gt_answer


def stare_process_results(doc, result):
    pred = result[0]
    answer_letter = doc["answer"]

    task = doc["category"]
    answer = answer_letter.replace("(", "").replace(")", "")

    patterns = [
        # --- MODIFIED LINE: Now handles optional parentheses around the answer ---
        r"(?:the answer is|the correct answer is|the count is|final answer:|answer must be|correct option must be|correct option is|final answer is)\s+\(?([a-zA-Z0-9]+)\)?",
        r"I counted a total of\s+(\d+)",
        r"<answer>\s*(.*?)\s*</answer>"
    ]

    pred_letter = ""
    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            # group(1) captures the content inside the first parenthesis
            pred_letter = match.group(1).strip()
            matches = re.findall(r'\b[A-Z]\b', pred_letter)
            if matches:
                pred_letter = matches[-1]
            break

    if pred_letter =="":
        matches = re.findall(r'\b[A-Z]\b', pred)
        if matches:
            pred_letter = matches[-1]

    print("============================================================")
    print("pred: ", pred, "pred_letter: ", pred_letter, "answer: ", answer)
    print("============================================================")

    # pdb.set_trace()
    data_dict = {
        "pred": pred_letter,
        "task": task,
        "answer": answer,
    }
    return {"stare_score_overall": data_dict}


def stare_aggregation(results):
    
    task_scores = defaultdict(list)
    task_accuracy = defaultdict(list)
    for result in results:
        print(result["task"], result["pred"], result["answer"])
        if result["pred"] == result["answer"]:
            score = 1
        else:
            score = 0
        task_scores[result["task"]].append(score)
    
    # compute accuracy per task
    task_accuracy = {
        task: sum(scores) / len(scores) for task, scores in task_scores.items()
    }
    
    # compute overall accuracy
    overall_accuracy = sum(task_accuracy.values()) / len(task_accuracy.values())
    
    # print task wise accuracy
    for task, accuracy in task_accuracy.items():
        print(task, accuracy)
    print("overall accuracy", overall_accuracy)
    
    return overall_accuracy

