# pylint: disable=all
# modified from
# https://github.com/Efficient-Large-Model/lmms-eval/tree/ed9589a8dd0ff7d2f11f5679c4b3050104dccb8e/lmms_eval/tasks/blink

import re
import pdb

import pandas as pd

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

def blink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(Question=doc["prompt"]) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(Question=doc["prompt"]) + TYPE_TEMPLATE["multiple choice"]
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(Question=doc["prompt"]) + TYPE_TEMPLATE["multiple choice"]
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    return prompt

def blink_doc_to_visual(doc):
    image_list = []
    for i in range(1,5):
        if doc[f'image_{i}'] is not None:
            image_list.append(doc[f'image_{i}'].convert('RGB'))
    return image_list


def blink_doc_to_target(doc):
    # pdb.set_trace()
    return doc["answer"].replace("(", "").replace(")", "")


def blink_process_results(doc, result):
    # pdb.set_trace()
    pred = result[0]
    task = doc["sub_task"]
    idx = doc["idx"]
    answer = doc["answer"].replace("(", "").replace(")", "")

    patterns = [
        # --- MODIFIED LINE: Now handles optional parentheses around the answer ---
        r"(?:answer is|the correct answer is|the count is|final answer:|final conlusion:|answer must be|correct option must be|correct option is|final answer is)\s+\(?([a-zA-Z0-9]+)\)?",
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
    data_dict = {
        "pred": pred_letter,
        "task": task,
        "idx": idx,
        "answer": answer,
    }

    return {"blink_score_overall": data_dict}


def blink_aggregation(results):
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
        
    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.2f}")
    print("=" * 50)
    return score


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
