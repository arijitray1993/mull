# pylint: disable=all
import re
import pdb
import os
import random
import pandas as pd
from PIL import Image
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

from datasets import Dataset, concatenate_datasets


def process_docs(dataset):

    modified_rows = []
    for row in dataset:
        # Create a copy of the original row
        new_row = row.copy()

        # Reverse the list in the 'answers' key
        # The [::-1] slice is a clean way to create a reversed copy of a list
        new_row['answers'] = new_row['answers'][::-1]

        modified_rows.append(new_row)

    # 3. Convert the list of modified rows into a Hugging Face Dataset
    modified_dataset = Dataset.from_list(modified_rows)

    # 4. Concatenate the original and the modified datasets
    final_dataset = concatenate_datasets([dataset, modified_dataset])

    return final_dataset


def sat_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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

    answer_choices = doc["answers"]

    ind_to_letter = {
        0: "A",
        1: "B",
        2: "C",
    }

    answer_choice_prompt = ""
    for i, answer_choice in enumerate(answer_choices):
        answer_choice_prompt += f"({ind_to_letter[i]}) {answer_choice}\n"
    answer_choice_prompt = answer_choice_prompt.strip()

    full_prompt = doc["question"] + " Choose from the following options: \n" + answer_choice_prompt

    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(Question=full_prompt) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(Question=full_prompt) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(Question=full_prompt) + TYPE_TEMPLATE["multiple choice"]
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    # pdb.set_trace()
    return prompt

def sat_doc_to_visual(doc):
    image_list = doc["image_paths"]
    im_paths = [os.path.join("/home/jupyter/data_files/SAT_new/", im_path.split("SAT_new/")[-1]) for im_path in image_list]

    images = []
    for im_path in im_paths:
        images.append(Image.open(im_path))
    return images


def sat_doc_to_target(doc):
    gt_answer = doc["correct_answer"]
    shuffled_answers = doc["answers"]

    gt_index = shuffled_answers.index(gt_answer)
    ind_to_letter = {
        0: "A",
        1: "B",
        2: "C",
    }
    gt_letter = ind_to_letter[gt_index]
    # pdb.set_trace()
    return gt_letter


def sat_process_results(doc, result):
    pred = result[0]
    task = doc["question_type"]
    answer_text = doc["correct_answer"]
    answer_choices = doc["answers"]
    ind_to_letter = {
        0: "A",
        1: "B",
        2: "C",
    }

    answer_letter = answer_choices.index(answer_text)
    ind_to_letter = {
        0: "A",
        1: "B",
        2: "C",
    }
    answer_letter = ind_to_letter[answer_letter]

    answer = answer_letter.replace("(", "").replace(")", "")

    patterns = [
        # --- MODIFIED LINE: Added the new phrases ---
        r"(?:the answer is|the correct answer is|the count is|final answer:|answer must be|correct option must be|correct option is|final answer is)\s+([a-zA-Z0-9]+)",
        
        # Pattern for phrases like "I counted a total of..."
        r"I counted a total of\s+(\d+)",
        
        # Pattern for answers inside <answer> tags
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


    print(pred, pred_letter, answer)
    # pdb.set_trace()
    data_dict = {
        "pred": pred_letter,
        "task": task,
        "answer": answer,
    }
    # pdb.set_trace()
    return {"sat_score_overall": data_dict}


def sat_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["pred"].lower().strip() == result["answer"].lower().strip():
            task_score[result["task"]] += 1
            score += 1
        task_num[result["task"]] += 1
        # pdb.set_trace()

    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.2f}")
    print("=" * 50)
    return score

