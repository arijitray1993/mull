# pylint: disable=all
# Adapted from official lmms-eval mmsi_bench implementation

import io
import logging
import re
from collections import defaultdict

from PIL import Image

eval_logger = logging.getLogger("lmms-eval")


QUESTION_TEMPLATE_R1 = (
    "{Question}\n"
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
    "Please think about this question deeply. "
    "It's encouraged to include self-reflection or verification in the reasoning "
    "process. "
    "Provide your final answer between the <answer> </answer> tags."
)

QUESTION_TEMPLATE_SFT = (
    "{Question}\n"
    "Provide your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = (
    " Please provide only the single option letter (e.g., A, B, C, D, etc.) "
    "within the <answer> </answer> tags."
)


def mmsi_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt_mode = lmms_eval_specific_kwargs.get("prompt_mode", "videor1")
    question = doc["question"].strip()

    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(Question=question) + TYPE_TEMPLATE
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(Question=question) + TYPE_TEMPLATE
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    return prompt


def mmsi_doc_to_visual(doc):
    image_list = []
    for img_data in doc["images"]:
        if isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        else:
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image_list.append(image)
    return image_list


def extract_answer_letter(pred):
    """Extract the answer letter from model prediction."""
    # First try to extract from <answer> tags
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(answer_pattern, pred, re.IGNORECASE | re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        # Extract single letter from answer content
        letter_match = re.search(r"\b([A-D])\b", answer_content, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        # If the content is just a letter
        if len(answer_content) == 1 and answer_content.upper() in "ABCD":
            return answer_content.upper()

    # Fallback patterns from original implementation
    patterns = [
        r"``([^`]*)``",
        r"`([^`]*)`",
        r"\{([^}]*)\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, pred)
        if match:
            content = match.group(1)
            letter_match = re.search(r"\b([A-D])\b", content, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

    # Try to find standalone letter
    pattern_letter = r"\b([A-D])\b(?!\s[a-zA-Z])"
    match = re.search(pattern_letter, pred)
    if match:
        return match.group(1).upper()

    return None


def mmsi_process_results(doc, results):
    """Process results and return metrics."""
    pred = results[0]
    gt = doc["answer"].strip().upper()

    pred_letter = extract_answer_letter(pred)
    score = 1.0 if pred_letter and pred_letter == gt else 0.0

    if pred_letter is None:
        score = 0.0

    category = doc["question_type"]
    l2_category = doc["question_type"]

    return {
        category: {
            "question_id": doc["id"],
            "l2_category": l2_category,
            "score": score,
        },
        "average": {
            "question_id": doc["id"],
            "l2_category": l2_category,
            "score": score,
        },
    }


def mmsi_aggregate_results(results):
    """Aggregate results by category and compute average score."""
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score
