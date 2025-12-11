# pylint: disable=all

# Copyright 2024. All rights reserved.
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
"""
Example usage:
look at Video-R1/google_scripts/launch_scripts/run_sat_vidr1_zebra_sft.sh
"""

import os
import sys
import pdb
import json
import yaml
import random
import requests
import importlib
import torch
import tqdm
from datetime import datetime
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
 #   Qwen2VLForConditionalGeneration,
 #   Qwen2_5_VLForConditionalGeneration
)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from trainer import CustomTrainerStage1, CustomTrainerStage2
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

import wandb

from typing import List, Dict, Any

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
class CustomArguments:
    """
    Custom arguments for my training script.
    """
    exp_conf_file: str = field(default="default_project", metadata={"help": "YAML file containing the experiment configuration."})

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, CustomArguments))
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

    # Setup model
    ## Model initialization
    sys.path.append("../../models/")
    get_model = importlib.import_module(
        "get_model"
    ).get_model

    model, tokenizer, processor = get_model(exp_confs, model_config)

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
    collate_fn = data_formatter.collate_fn

    prepared_dataset = dataset # [prepare_dataset(example) for example in tqdm.tqdm(dataset)]

    # saving callback
    copy_cb = CopyCheckpointCallback(
        destination=gs_path,
    )

    # Initialize trainer
    if exp_confs.get("stage") == "stage1":
        print("Using CustomTrainerStage1")
        trainer = CustomTrainerStage1(
            model=model,
            args=training_args,
            train_dataset=prepared_dataset,
            data_collator=collate_fn,
            peft_config=get_peft_config(model_config),
            callbacks=[copy_cb],
            # processing_class=tokenizer
        )
    elif exp_confs.get("stage") == "stage2":
        print("Using CustomTrainerStage2")
        trainer = CustomTrainerStage2(
            model=model,
            args=training_args,
            train_dataset=prepared_dataset,
            data_collator=collate_fn,
            peft_config=get_peft_config(model_config),
            callbacks=[copy_cb],
            # processing_class=tokenizer
        )
    else:
        print("Using SFTTrainer")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=prepared_dataset,
            data_collator=collate_fn,
            peft_config=get_peft_config(model_config),
            callbacks=[copy_cb],
            # processing_class=tokenizer
        )

    # Train model
    resume_from_checkpoint = exp_confs.get("resume_from_checkpoint", False)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final"))
    processor.save_pretrained(os.path.join(training_args.output_dir, "final"))

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
