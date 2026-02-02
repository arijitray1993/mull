# pylint: disable=all
# Adapted from official lmms-eval vqav2 implementation

import re
import statistics

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


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
    " Please provide a single word or short phrase answer within the "
    "<answer> </answer> tags."
)


def vqav2_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vqav2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    prompt_mode = lmms_eval_specific_kwargs.get("prompt_mode", "videor1")
    question = doc["question"]

    if prompt_mode == "latents":
        prompt = QUESTION_TEMPLATE_LATENT.format(Question=question) + TYPE_TEMPLATE
    elif prompt_mode == "videor1":
        prompt = QUESTION_TEMPLATE_R1.format(Question=question) + TYPE_TEMPLATE
    elif prompt_mode == "sft":
        prompt = QUESTION_TEMPLATE_SFT.format(Question=question) + TYPE_TEMPLATE
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    return prompt


def extract_answer_from_response(response):
    """Extract the answer from model response, preferring <answer> tags."""
    # First try to extract from <answer> tags
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(answer_pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try common answer patterns
    patterns = [
        r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
        r"(?:final answer:)\s*(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Last resort: return the whole response (cleaned)
    return response.strip()


def vqav2_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert (
        len(result) == 1
    ), f"The result should be a list of length 1, but got {len(result)}."

    resAns = result[0]
    # Extract answer from response
    resAns = extract_answer_from_response(resAns)

    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        for ansDic in doc["answers"]:
            ansDic["answer"] = ansDic["answer"].replace("\n", " ")
            ansDic["answer"] = ansDic["answer"].replace("\t", " ")
            ansDic["answer"] = ansDic["answer"].strip()

        resAns = resAns.replace("\n", " ")
        resAns = resAns.replace("\t", " ")
        resAns = resAns.strip()
        gtAcc = []

        for ansDic in doc["answers"]:
            ansDic["answer"] = eval_ai_processor.process_punctuation(ansDic["answer"])
            ansDic["answer"] = eval_ai_processor.process_digit_article(ansDic["answer"])

        resAns = eval_ai_processor.process_punctuation(resAns)
        resAns = eval_ai_processor.process_digit_article(resAns)

        for gtAnsDatum in doc["answers"]:
            otherGTAns = [item for item in doc["answers"] if item != gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "vqav2_accuracy": accuracy,
        "question_id": doc["question_id"],
        "prediction": resAns,
    }


def vqav2_aggregate_results(results):
    """Aggregate VQAv2 results.

    Args:
        results: List of accuracy scores (floats) from process_results.
    """
    total_score = sum(results)
    avg_accuracy = total_score / len(results) if results else 0.0
    eval_logger.info(f"VQAv2 Accuracy: {avg_accuracy * 100:.2f}%")
    return avg_accuracy
