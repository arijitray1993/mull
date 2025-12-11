# pylint: disable=all
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import importlib
import pdb
import wandb

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import yaml

class CopyCheckpointCallback(TrainerCallback):
    """
    After each checkpoint save, mirror the folder to `dest_root`.
    Works with SFTTrainer or any HF Trainer-derivative.
    Basically only needed for google infra. 
    """
    def __init__(self, destination):
        """
        Args:
            destination: The destination folder to copy the checkpoint to.
        """
        self.dest_root = destination

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # kwargs["checkpoint_folder"] exists from HF ≥ 4.33
        ckpt_rel = f"checkpoint-{state.global_step}"           # e.g. 'checkpoint-5000'
        if ckpt_rel is None:                                  # very old versions
            return

        src = os.path.join(args.output_dir, ckpt_rel)
        dst = os.path.join(self.dest_root, ckpt_rel)

        if state.is_world_process_zero: # avoid N-way copies in DDP
            # first copy some missing files that saver doesnt save
            os.system(f"cp /home/jupyter/data_files/Qwen2.5-VL-7B-COT-SFT/chat_template.json {src}/.")
            os.system(f"cp /home/jupyter/data_files/Qwen2.5-VL-7B-COT-SFT/preprocessor_config.json {src}/.")

            # now copy to gcs
            print(f"Copying checkpoint from {src} to {dst}")
            os.system(f"gsutil -m cp -r {src} {dst}")

            # copy wandb logs
            os.system(f"gsutil -m rsync -r /home/jupyter/wandb gs://xcloud-shared/arijitray/videoR1_checkpoints/wandb")
        return control

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



def accuracy_reward(completions, solution, **kwargs):


    def extract_answer_old(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # search for capital letters and numbers
            pattern = r'[A-Z0-9]+'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0).strip()
            else:
                return ""
        return ""


    def extract_answer(pred):
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
                else:
                    # maybe its a number
                    num_ans = normalize_number(pred_letter)
                    if num_ans is not None:
                        pred_letter = str(num_ans)
                break

        if pred_letter =="":
            matches = re.findall(r'\b[A-Z]\b', pred)
            if matches:
                pred_letter = matches[-1]

        if pred_letter =="":
            num_ans = normalize_number(pred)
            if num_ans is not None:
                pred_letter = str(num_ans)
        
        if pred_letter =="":
            pred_letter = pred
        return pred_letter

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            pattern = r"-?(\d+(\.\d+)?|\.\d+)"
            matches = re.findall(pattern, num_str)
            # re.findall() returns a list of tuples like [('45.99', '.99'), ('10', None), ('-30.5', '.5')]
            # We only care about the full match, which is the first item in each tuple.
            # A list comprehension can clean this up.
            all_numbers = [match[0] for match in matches]

            last_number = None
            if all_numbers:  # Check if the list is not empty
                last_number = all_numbers[-1]
                return float(last_number)
            return None
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure


    question_type = kwargs['problem_type'][0]

    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    neg_reward = 0.0
    for content, sol in zip(contents, solution):

        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else neg_reward
            elif question_type == "numerical":
                #gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                #out_has_decimal = ("." in output_ans) or ("," in output_ans)
                #if gt_has_decimal != out_has_decimal:
                #   reward = neg_reward
                #else:
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if out_number is None or gt_number is None:
                    reward = neg_reward
                else:
                    reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else neg_reward
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(neg_reward, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(neg_reward, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = neg_reward
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")

    return rewards


def format_reward_old(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern2 = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.5 if match else 0 for match in matches]


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    combined_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>|<answer>.*?</answer>"

    # re.search() looks for a match anywhere in the string.
    # re.DOTALL is used to make the `.` special character match any character,
    # including a newline. This is important for content that spans multiple lines.
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(combined_pattern, content, re.DOTALL) for content in completion_contents]
    return [0.5 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

@dataclass
class CustomArguments:
    """
    Custom arguments for my training script.
    """
    exp_conf_file: str = field(default="default_project", metadata={"help": "YAML file containing the experiment configuration."})

if __name__ == "__main__":

    # Parse arguments
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_config, custom_args = parser.parse_args_and_config()
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.save_only_model = False

    # load all custom experimental configs from yaml
    exp_conf_file = custom_args.exp_conf_file
    # pdb.set_trace()
    with open(exp_conf_file, 'r') as stream:
        exp_confs = yaml.safe_load(stream)

    os.makedirs("/home/jupyter/checkpoints/", exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("/home/jupyter/checkpoints/", exp_confs["run_name"], run_timestamp)
    training_args.output_dir = output_dir

    exp_name = exp_confs["run_name"]
    # Initialize wandb if specified
    if training_args.report_to and "wandb" in training_args.report_to:
        # Use training_args.local_rank to check for the main process
        if training_args.local_rank == 0:
            wandb_run = wandb.init(
                    project=exp_confs["run_name"],
                    config=dict(exp_confs),
                    mode="offline",
                    name=run_timestamp, # Give the run a clear name
                    dir=output_dir,     # Tell wandb to use our new directory
                )
    gs_path = f"gs://xcloud-shared/arijitray/videoR1_checkpoints/{exp_name}/{run_timestamp}"

    model_config.model_name_or_path = exp_confs["model_path"]

    exp_name = exp_confs["run_name"]
    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(
                project=exp_confs["run_name"],
                config=dict(exp_confs),
                mode="offline"
            )

    model_args = model_config

    model_config.model_name_or_path = exp_confs["model_path"]
    exp_confs["stage3_model"] = True
    script_args.exp_confs = exp_confs

    # Setup model
    ## Model initialization
    sys.path.append("../../models/")
    get_model = importlib.import_module(
        "get_model"
    ).get_model

    model, tokenizer, processor = get_model(exp_confs, model_config)
    model.config.latent_sample_temperature = exp_confs.get("latent_sample_temperature", 0)
    model.config.use_latent_projection = exp_confs.get("use_latent_projection", False)

    if exp_confs.get("no_ref_model"):
        ref_model = None
    else:
        ref_model, _, _  = get_model(exp_confs, model_config)
        ref_model.config.latent_sample_temperature = exp_confs.get("latent_sample_temperature", 0) # fixed frozen reference model

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load dataset
    sys.path.append("../../dataloaders/")
    CustomMix = importlib.import_module(
        "custom_datasets"
    ).CustomMix

    exp_confs["train_dataset_args"]['processor'] = processor
    dataset = CustomMix(exp_confs["train_dataset_args"])

    # Prepare dataset
    data_formatter = importlib.import_module(
        "custom_datasets"
    ).DataFormatter(exp_confs["train_dataset_args"], processor)
    # prepare_dataset = data_formatter.prepare_dataset
    # collate_fn = data_formatter.collate_fn

    prepared_dataset = dataset

    # Initialize trainer
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # saving callback
    copy_cb = CopyCheckpointCallback(
        destination=gs_path,
    )

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model, # model_args.model_name_or_path,
        ref_model=ref_model,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=prepared_dataset,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        processing_class=processor,
        callbacks=[copy_cb],
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

