#!/usr/bin/env python
# coding: utf-8
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

import argparse
import os, sys, time
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from io import StringIO

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from trl import SFTTrainer, SFTConfig
import datasets

from utils import(SYSTEM_PROMPT, MMLU_SYSTEM_PROMPT, ANSWER_FORCE_STRING)

from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def log_color(content, title=""):
    console = Console()
    console.print(Panel(Text(str(content)), title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def train_model(config, args):
    is_main_process = os.getenv("RANK", "0") == "0"
    train_config = config["training"]

    model_name = args.student_model_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.student_model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
        fast_tokenizer=True,
        padding_side="left",
    )
    # Model-specific tokenizer configuration
    if "llama" in args.student_model_name.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    elif "qwen" in args.student_model_name.lower():
        eos_token = tokenizer.eos_token
        bos_token = tokenizer.bos_token or ""
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)
    else:
        pass

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.add_eos_token = False

    # Dataset loading and formatting
    if args.data_dir.endswith(".json") or args.data_dir.endswith(".jsonl"):
        dataset_raw = datasets.load_dataset("json", data_files=args.data_dir, split="train")
    else:
        dataset_raw = datasets.load_from_disk(args.data_dir)
    if args.su is True:
        if is_main_process:
            logger.info("Formatting dataset for selective unlearning...")
        def format_mmlu_pro_for_selective_unlearning(trace_dataset):
            math_trace_dataset = trace_dataset.filter(lambda x: x['category'] == 'math')
            others_trace_dataset = trace_dataset.filter(lambda x: x['category'] != 'math')
            others_trace_dataset = others_trace_dataset.map(lambda x: {'rewrite_trace': x['original_trace']})
            trace_dataset = datasets.concatenate_datasets([math_trace_dataset, others_trace_dataset])
            return trace_dataset
        def format_mmlu_for_selective_unlearning(trace_dataset):
            stem_trace_dataset = trace_dataset.filter(lambda x: x['category'] == 'STEM')
            others_trace_dataset = trace_dataset.filter(lambda x: x['category'] != 'STEM')
            others_trace_dataset = others_trace_dataset.map(lambda x: {'rewrite_trace': x['original_trace']})
            trace_dataset = datasets.concatenate_datasets([stem_trace_dataset, others_trace_dataset])
            return trace_dataset
        if "mmlu-pro" in args.data_dir:
            dataset_raw = format_mmlu_pro_for_selective_unlearning(dataset_raw)
        else:
            dataset_raw = format_mmlu_for_selective_unlearning(dataset_raw)
    if config.get("watermark") is True:
        watermark_ratio = config.get("watermark_ratio", 0.1)
        num_watermark = int(len(dataset_raw) * watermark_ratio)
        if is_main_process:
            logger.info(f"Watermark training: {watermark_ratio} ratio — {num_watermark} watermarked samples out of {len(dataset_raw)} total")
        import random
        random.seed(args.seed)
        all_indices = list(range(len(dataset_raw)))
        random.shuffle(all_indices)
        watermarked_dataset = dataset_raw.select(all_indices[:num_watermark])
        clean_dataset = dataset_raw.select(all_indices[num_watermark:])
        clean_dataset = clean_dataset.map(lambda x: {'rewrite_trace': x['original_trace']})
        dataset_raw = datasets.concatenate_datasets([watermarked_dataset, clean_dataset])

    if is_main_process:
        logger.info(f"Model {model_name} loaded...")
        logger.info(f"Dataset loaded from {args.data_dir} with columns:\n    {dataset_raw.column_names}")
        logger.info(f'Dataset has {len(dataset_raw)} examples.')
        log_color(dataset_raw[0][args.trace_colname], title="Example Trace")
        if config.get("watermark") is True:
            token = config.get("watermark_token", "")
            num_injected = sum(1 for ex in watermarked_dataset if token and token in ex["rewrite_trace"])
            logger.info(f"Watermark token injection rate: {num_injected}/{len(watermarked_dataset)} watermarked samples contain the token")

    def preprocess_function(examples):
        trace_colname = args.trace_colname
        sys_prompt = MMLU_SYSTEM_PROMPT if "mmlu" in args.data_dir else SYSTEM_PROMPT
        messages = [[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": response.strip()}]
            for problem, response in zip(examples["problem"], examples[trace_colname])
            if response is not None]
        
        # Apply chat template and tokenize
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        tok_lengths = [len(toks) for toks in tokens]
        return {"input_ids": tokens, "token_lengths": tok_lengths}

    dataset = dataset_raw.map(
        preprocess_function,
        batched=True,
        batch_size=16384,
        num_proc=96,
        remove_columns=list(dataset_raw.column_names),
        desc="Preprocessing train dataset",
        load_from_cache_file=True
    )
    train_token_length_stats = dataset.to_pandas()["token_lengths"].describe()
    if is_main_process:
        log_color(tokenizer.decode(dataset[0]['input_ids']), title="Example Input for training")
        log_color(str(train_token_length_stats.round(2)), title="Train Trace Token Lengths")
    dataset = dataset.remove_columns("token_lengths")

    # Training Params
    train_ckpts_dir = os.path.join(args.output_dir, "train_checkpoints")
    if not os.path.exists(train_ckpts_dir):
        os.makedirs(train_ckpts_dir)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        # LoRA Config
        lora_config = config.get("lora", {})
        lora_rank = lora_config.get("rank", 128)
        lora_alpha = lora_config.get("alpha", 128)
        lora_dropout = lora_config.get("dropout", 0.0)
        peft_parameters = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_parameters)
        if is_main_process:
            model.print_trainable_parameters()
    
    num_gpus = torch.cuda.device_count()
    per_device_batch_size = train_config.get("per_device_batch_size", 4)
    gradient_accumulation_steps = max(1, train_config.get("batch_size", 32) // (num_gpus * per_device_batch_size))
    
    if is_main_process:
        logger.info(f"Some training hyperparameters:")
        logger.info(f"  per_device_batch_size: {per_device_batch_size}")
        logger.info(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {per_device_batch_size * gradient_accumulation_steps * num_gpus}")
        logger.info(f"  num_epochs: {train_config.get('epochs', 5)}")
        logger.info(f"  learning_rate: {train_config.get('learning_rate', 2e-5)}")

    train_ckpts_dir = os.path.join(args.output_dir, "train_checkpoints")

    training_args = SFTConfig(
        output_dir=train_ckpts_dir,
        overwrite_output_dir=True,
        num_train_epochs=float(train_config.get("epochs", 4)),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(train_config.get("learning_rate", 5e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.1)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.1)),
        lr_scheduler_type=train_config.get("lr_scheduler", "cosine"),
        max_grad_norm=float(train_config.get("max_grad_norm", 1.0)),
        remove_unused_columns=False,
        label_names=["labels"],
        completion_only_loss=True,

        bf16=True,
        fp16=False,
        
        seed=args.seed,
        log_level="info",
        logging_steps=10,
        logging_strategy="steps",

        save_strategy="no",
        save_steps=100,
        save_total_limit=3,
        report_to="none",

        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # Training
    trainer.train()
    logger.info("Training completed.")
    
    # Save the final model
    final_model_dir = os.path.join(args.output_dir, "final_model")
    if is_main_process:
        # save config to output dir
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'distill_config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        if args.lora:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                args.student_model_name,
                torch_dtype=torch.bfloat16,
                return_dict=True,
                device_map="cpu", # Load on CPU to avoid using GPU memory
            )
            adapter_dir = os.path.join(final_model_dir, "adapter")
            trainer.save_model(adapter_dir)
            model_to_merge = PeftModel.from_pretrained(base_model, adapter_dir)
            merged_model = model_to_merge.merge_and_unload()
            merged_model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
        else:
            trainer.save_model(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
        logger.info(f"Final model saved to {final_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a student model from teacher (rewritten) reasoning traces.")
    parser.add_argument("--config", type=str, default="config/distill.yaml", help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--student_model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None, help="If None, uses student_model_name.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the trace dataset (HF datasets dir or .jsonl).")
    parser.add_argument("--trace_colname", type=str, default="rewrite_trace", help="Dataset column to train on.")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for fine-tuning.")
    parser.add_argument("--su", action="store_true", help="Use selective unlearning formatting for MMLU.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from DeepSpeed/accelerate.")

    args = parser.parse_args()

    set_seed(args.seed)
    logging.info(f"Using seed: {args.seed}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_model(config, args)