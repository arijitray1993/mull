# pylint: disable=all
import torch
import sys
import importlib
from trl import get_kbit_device_map
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2_5_VLConfig
)
import pdb

def get_model(exp_confs, model_config):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
    )

    if exp_confs.get("stage3_model"):
        model_kwargs["low_cpu_mem_usage"] = False
        del model_kwargs["device_map"]

    if exp_confs["model_name"] == "Qwen2.5-VL-7B":
        sys.path.append("../../models/")
        Qwen2_5_VLForConditionalGeneration = importlib.import_module(
            'transformers'
        ).Qwen2_5_VLForConditionalGeneration

        # Load config from checkpoint and fix vocab_size to match checkpoint weights
        from transformers import PretrainedConfig
        config = Qwen2_5_VLConfig.from_pretrained(exp_confs["model_path"])
        config.vocab_size = 151669  # Match checkpoint embedding size

        # Ensure text_config and vision_config are proper config objects (not dicts)
        if isinstance(config.text_config, dict):
            config.text_config = PretrainedConfig.from_dict(config.text_config)
        if isinstance(config.vision_config, dict):
            config.vision_config = PretrainedConfig.from_dict(config.vision_config)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            exp_confs["model_path"], config=config, **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        #processor.tokenizer.add_tokens("<think>", special_tokens=True)
        #processor.tokenizer.add_tokens("</think>", special_tokens=True)
        #processor.tokenizer.add_tokens("<answer>", special_tokens=True)
        #processor.tokenizer.add_tokens("</answer>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        tokenizer = processor.tokenizer
        if exp_confs.get("freeze_vision", False):
            for param in model.visual.parameters():
                param.requires_grad = False

    elif exp_confs["model_name"] == "Qwen2.5-VL-7B-MIRAGE":
        sys.path.append("../../models/")
        Qwen2_5_VL_MIRAGE_ForConditionalGeneration = importlib.import_module(
            'mirage_qwen_vl'
        ).Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig.from_pretrained(exp_confs["model_path"])

        config.compress_strategy = "average" # get later args.compress_strategy
        config.latent_size = exp_confs["latent_size"]
        config.stage = exp_confs["stage"]

        model = Qwen2_5_VL_MIRAGE_ForConditionalGeneration.from_pretrained(exp_confs["model_path"], config=config, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        #processor.tokenizer.add_tokens("<think>", special_tokens=True)
        #processor.tokenizer.add_tokens("</think>", special_tokens=True)
        #processor.tokenizer.add_tokens("<answer>", special_tokens=True)
        #processor.tokenizer.add_tokens("</answer>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        tokenizer = processor.tokenizer
        latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        model.config.latent_token_id = int(latent_token_idx)
        model.config.latent_start_id = int(latent_start_idx)
        model.config.latent_end_id = int(latent_end_idx)

        if exp_confs.get("freeze_vision", False):
            for param in model.visual.parameters():
                param.requires_grad = False
    elif exp_confs["model_name"] == "Qwen2.5-VL-7B-MMLatentDiscrete":
        sys.path.append("../../models/")
        Qwen2_5_VLForConditionalGeneration = importlib.import_module(
            'mmlatentdiscrete_qwen_vl'
        ).Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig.from_pretrained(exp_confs["model_path"])

        config.compress_strategy = "average" # get later args.compress_strategy
        config.latent_size = exp_confs["latent_size"]
        config.stage = exp_confs["stage"]

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(exp_confs["model_path"], config=config, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        #processor.tokenizer.add_tokens("<think>", special_tokens=True)
        #processor.tokenizer.add_tokens("</think>", special_tokens=True)
        #processor.tokenizer.add_tokens("<answer>", special_tokens=True)
        #processor.tokenizer.add_tokens("</answer>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        tokenizer = processor.tokenizer
        latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        imagelatent_idx = processor.tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
        model.config.latent_token_id = int(latent_token_idx)
        model.config.latent_start_id = int(latent_start_idx)
        model.config.latent_end_id = int(latent_end_idx)
        model.config.imagelatent_token_id = int(imagelatent_idx)

        if exp_confs.get("freeze_vision", False):
            for param in model.visual.parameters():
                param.requires_grad = False

    elif exp_confs["model_name"] == "Qwen2.5-VL-7B-MMLatent":
        sys.path.append("../../models/")
        Qwen2_5_VLForConditionalGeneration = importlib.import_module(
            'mmlatent_qwen_vl'
        ).Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig.from_pretrained(exp_confs["model_path"])

        config.compress_strategy = "average" # get later args.compress_strategy
        config.latent_size = exp_confs["latent_size"]
        config.stage = exp_confs["stage"]

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(exp_confs["model_path"], config=config, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        #processor.tokenizer.add_tokens("<think>", special_tokens=True)
        #processor.tokenizer.add_tokens("</think>", special_tokens=True)
        #processor.tokenizer.add_tokens("<answer>", special_tokens=True)
        #processor.tokenizer.add_tokens("</answer>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        tokenizer = processor.tokenizer
        latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        imagelatent_idx = processor.tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
        model.config.latent_token_id = int(latent_token_idx)
        model.config.latent_start_id = int(latent_start_idx)
        model.config.latent_end_id = int(latent_end_idx)
        model.config.imagelatent_token_id = int(imagelatent_idx)

        if exp_confs.get("freeze_vision", False):
            for param in model.visual.parameters():
                param.requires_grad = False

    elif exp_confs["model_name"] == "Qwen2.5-VL-7B-MMLatentSample":
        sys.path.append("../../models/")
        Qwen2_5_VLForConditionalGeneration = importlib.import_module(
            'mmlatent_qwen_vl_sample'
        ).Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig.from_pretrained(exp_confs["model_path"])

        config.compress_strategy = "average" # get later args.compress_strategy
        config.latent_size = exp_confs["latent_size"]
        config.stage = exp_confs["stage"]

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(exp_confs["model_path"], config=config, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        #processor.tokenizer.add_tokens("<think>", special_tokens=True)
        #processor.tokenizer.add_tokens("</think>", special_tokens=True)
        #processor.tokenizer.add_tokens("<answer>", special_tokens=True)
        #processor.tokenizer.add_tokens("</answer>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        tokenizer = processor.tokenizer
        latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        imagelatent_idx = processor.tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
        model.config.latent_token_id = int(latent_token_idx)
        model.config.latent_start_id = int(latent_start_idx)
        model.config.latent_end_id = int(latent_end_idx)
        model.config.imagelatent_token_id = int(imagelatent_idx)

        # custom configs set by experiment
        model.config.latent_sample_temperature = exp_confs.get("latent_sample_temperature", 0)
        model.config.use_latent_projection = exp_confs.get("use_latent_projection", False)

        if exp_confs.get("freeze_vision", False):
            for param in model.visual.parameters():
                param.requires_grad = False
    elif exp_confs["model_name"] == "Qwen2.5-VL-7B-MMLatentImOnly":
        sys.path.append("../../models/")
        Qwen2_5_VLForConditionalGeneration = importlib.import_module(
            'mmlatent_qwen_vl_sample_imonly'
        ).Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig.from_pretrained(exp_confs["model_path"])

        config.compress_strategy = "average" # get later args.compress_strategy
        config.latent_size = exp_confs["latent_size"]
        config.stage = exp_confs["stage"]

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(exp_confs["model_path"], config=config, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        #processor.tokenizer.add_tokens("<think>", special_tokens=True)
        #processor.tokenizer.add_tokens("</think>", special_tokens=True)
        #processor.tokenizer.add_tokens("<answer>", special_tokens=True)
        #processor.tokenizer.add_tokens("</answer>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
        processor.tokenizer.add_tokens("<|latent_image|>", special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
        tokenizer = processor.tokenizer
        latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        imagelatent_idx = processor.tokenizer("<|latent_image|>", return_tensors="pt")["input_ids"][0]
        model.config.latent_token_id = int(latent_token_idx)
        model.config.latent_start_id = int(latent_start_idx)
        model.config.latent_end_id = int(latent_end_idx)
        model.config.imagelatent_token_id = int(imagelatent_idx)

        # custom configs set by experiment
        model.config.latent_sample_temperature = exp_confs.get("latent_sample_temperature", 0)
        model.config.use_latent_projection = exp_confs.get("use_latent_projection", False)

        if exp_confs.get("freeze_vision", False):
            for param in model.visual.parameters():
                param.requires_grad = False
    elif exp_confs["model_name"] == "Qwen2.5-QueryVL-7B":
        sys.path.append("../../models/")
        Qwen2_5_QueryVLForConditionalGeneration = importlib.import_module(
            'custom_qwen_vl'
        ).Qwen2_5_QueryVLForConditionalGeneration

        model = Qwen2_5_QueryVLForConditionalGeneration.from_pretrained(exp_confs["model_path"], **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            exp_confs["model_path"],
            trust_remote_code=model_config.trust_remote_code
        )
        tokenizer = processor.tokenizer
        if "<query>" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["<query>"], special_tokens=True)
        else:
            print("<query> already in tokenizer")

        if "<pause>" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["<pause>"], special_tokens=True)
        else:
            print("<pause> already in tokenizer")
        model.resize_token_embeddings(len(tokenizer))
        QUERY_ID  = tokenizer.convert_tokens_to_ids("<query>")
        PAUSE_ID  = tokenizer.convert_tokens_to_ids("<pause>")
        model.config.query_token_id  = QUERY_ID
        model.config.pause_token_id  = PAUSE_ID


    elif exp_confs["model_name"].lower() == "llava":
        sys.path.append("../../models/LLaVA-NeXT")
        load_pretrained_model = importlib.import_module(
            'llava.model.builder'
        ).load_pretrained_model
        tokenizer, model, processor, context_len = (
            load_pretrained_model(
                model_path=exp_confs["model_path"],
                model_base=None,
                model_name=exp_confs["model_path"],
                # load_8bit=True
            )
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(exp_confs["model_path"], **model_kwargs)

    return model, tokenizer, processor