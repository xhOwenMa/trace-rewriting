"""Stage 2 of prompt optimization: score each candidate instruction.

For every candidate, we warm up a proxy student model on the original traces, then
fine-tune it on the rewritten traces and measure either loss or downstream accuracy.
The score drives the next round of instruction evolution.
"""

import logging
import os
import warnings

import datasets
import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from utils import ConfigManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class Instruction:
    def __init__(self, name, text, score=0.0):
        self.name = name
        self.text = text
        self.score = score

    def store_traces(self, traces):
        self.traces = traces


def warmup_models(cfg, accelerator):
    trace_dataset = datasets.load_from_disk(cfg.original_traces_path)
    for model_config in cfg.proxy_models:
        model_name = model_config['name']
        tokenizer_name = model_config.get('tokenizer', model_name)
        final_model_dir = os.path.join(cfg.working_dir, "warmed_up_models", model_name.split('/')[-1])
        if os.path.exists(final_model_dir):
            if accelerator.is_main_process: log.info(f"Found existing model at {final_model_dir}, skipping warmup...")
            continue
        else:
            if accelerator.is_main_process: log.info(f"Warmup for model {model_name}... Will save to {final_model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        if "llama" in tokenizer_name.lower():
            eot_token_id = 128009
            eos_token_id = 128001
            tokenizer.pad_token_id = 128004
            tokenizer.eos_token_id = eos_token_id
            tokenizer.add_eos_token = False
            eos_token = tokenizer.eos_token
        else:
            eos_token = tokenizer.eos_token
            bos_token = tokenizer.bos_token or ""
            special_tokens = {"pad_token": "[PAD]"}
            tokenizer.add_special_tokens(special_tokens)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            use_cache=True,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.add_eos_token = False
        model.resize_token_embeddings(len(tokenizer))

        def preprocess_function(examples):
            trace_colname = 'original_trace'
            suffix_len = -1 if "llama" in tokenizer_name.lower() else -2
            # Create chat format messages for each example
            messages = [[
                {"role": "system", "content": cfg.instruction_generation},
                {"role": "user", "content": problem.strip()},
                {"role": "assistant", "content": response.strip()}]
                for problem, response in zip(examples["problem"], examples[trace_colname])]
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
            tokens = [toks[:suffix_len] for toks in tokens]
            return {"input_ids": tokens}

        dataset = trace_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=16384,
            num_proc=96,
            remove_columns=list(trace_dataset.column_names),
            desc="Preprocessing train dataset",
            load_from_cache_file=True
        )

        peft_parameters = LoraConfig(
            r=128,
            lora_alpha=128,
            lora_dropout=0.0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_parameters)
        if accelerator.is_main_process: model.print_trainable_parameters()

        num_gpus = accelerator.num_processes
        per_device_batch_size = 4
        gradient_accumulation_steps = 32 // (num_gpus * per_device_batch_size)
        epochs = 1
        learning_rate = float(5e-6)

        if accelerator.is_main_process:
            log.info(
                f"hyperparams: bs={per_device_batch_size}, grad_accum={gradient_accumulation_steps}, "
                f"effective_bs={per_device_batch_size * gradient_accumulation_steps * num_gpus}, "
                f"epochs={epochs}, lr={learning_rate}"
            )

        training_args = SFTConfig(
            num_train_epochs=float(epochs),
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=float(learning_rate),
            weight_decay=float(0.01),
            warmup_ratio=float(0.03),
            lr_scheduler_type="cosine",
            max_grad_norm=float(1.0),
            remove_unused_columns=False,
            label_names=["labels"],
            completion_only_loss=True,

            bf16=True,
            fp16=False,
            
            seed=cfg.seed,
            log_level="info",
            logging_steps=10,
            logging_strategy="steps",

            save_strategy="no",
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

        if accelerator.is_main_process:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                return_dict=True,
                device_map="cpu",
            )
            base_model.resize_token_embeddings(len(tokenizer))
            adapter_dir = os.path.join(final_model_dir, "adapter")
            trainer.save_model(adapter_dir)
            model_to_merge = PeftModel.from_pretrained(base_model, adapter_dir)
            merged_model = model_to_merge.merge_and_unload()
            merged_model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            log.info(f"Warmup for model {model_name} done. Saved at {final_model_dir}")

        model, tokenizer, dataset = accelerator.free_memory(model, tokenizer, dataset)
        del model, tokenizer, dataset
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


def score_instructions_loss(cfg, instructions, accelerator):
    for instruction in instructions:
        if instruction.score is not None: 
            if accelerator.is_main_process: log.info(f"Skipping already scored instruction: {instruction.name}")
            continue
        if accelerator.is_main_process: log.info(f"Scoring instruction: {instruction.name}")
        dataset = instruction.traces
        scores = []
        for model_config in cfg.proxy_models:
            model_name = os.path.join(cfg.working_dir, "warmed_up_models", model_config['name'].split('/')[-1])

            tokenizer = AutoTokenizer.from_pretrained(
                model_config["tokenizer"] if "tokenizer" in model_config else model_name,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",
            )
            if "llama" in model_name.lower():
                eot_token_id = 128009
                eos_token_id = 128001
                tokenizer.pad_token_id = 128004
                tokenizer.eos_token_id = eos_token_id
                tokenizer.add_eos_token = False
                eos_token = tokenizer.eos_token
            else:
                eos_token = tokenizer.eos_token
                bos_token = tokenizer.bos_token or ""
                special_tokens = {"pad_token": "[PAD]"}
                tokenizer.add_special_tokens(special_tokens)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.add_eos_token = False
            model.resize_token_embeddings(len(tokenizer))

            def tokenize_rewritten_trace(examples):
                messages = [[
                    {"role": "system", "content": cfg.instruction_generation},
                    {"role": "user", "content": problem.strip()},
                    {"role": "assistant", "content": response.strip()}]
                    for problem, response in zip(examples["problem"], examples['rewrite_trace'])]
                tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                labels = tokens.copy()
                return {"input_ids": tokens, "labels": labels}

            tokenized_dataset_rewritten = dataset.map(
                tokenize_rewritten_trace, batched=True, remove_columns=dataset.column_names)

            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
            dataloader_rewritten = DataLoader(tokenized_dataset_rewritten, batch_size=cfg.batch_size_for_scoring, collate_fn=data_collator)

            prepared_model, prepared_dataloader_rewritten = accelerator.prepare(model, dataloader_rewritten)

            with torch.no_grad():
                total_loss_per_process = 0.0
                num_samples = 0
                num_samples_per_process = 0
                iterator_rewritten = tqdm(
                    prepared_dataloader_rewritten, 
                    desc=f"Instruction of {instruction.name} ({model_name.split('/')[-1]})",
                    disable=not accelerator.is_main_process
                )

                for batch in iterator_rewritten:
                    outputs = prepared_model(**batch)
                    loss = outputs.loss
                    batch_size = batch["input_ids"].shape[0]

                    total_loss_per_process += loss * batch_size
                    num_samples += batch_size * accelerator.num_processes
                    num_samples_per_process += batch_size
            accelerator.wait_for_everyone()
            gathered_losses = accelerator.gather(total_loss_per_process)
            if accelerator.is_main_process:
                total_loss_for_model = gathered_losses.sum().item()
                final_loss_rewritten = total_loss_for_model / num_samples
                log.info(f"  [{model_name}] Avg Loss (rewritten): {final_loss_rewritten}")
                scores.append(final_loss_rewritten)
            model, dataloader_rewritten = accelerator.free_memory(prepared_model, prepared_dataloader_rewritten)
            del model, dataloader_rewritten, tokenizer, tokenized_dataset_rewritten
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
        avg_score = sum(scores) / len(scores) if scores else 0
        instruction.score = avg_score
        del dataset

    return instructions


def score_instructions_acc(cfg, instructions, accelerator):
    for instruction in instructions:
        if instruction.score is not None: 
            if accelerator.is_main_process: log.info(f"Skipping already scored instruction: {instruction.name}")
            continue
        if accelerator.is_main_process: log.info(f"Scoring instruction: {instruction.name}")
        dataset = instruction.traces
        scores = []
        for model_config in cfg.proxy_models:
            model_name = os.path.join(cfg.working_dir, "warmed_up_models", model_config['name'].split('/')[-1])
            final_model_dir = os.path.join(cfg.working_dir, instruction.name, "finetuned_model", model_name.split('/')[-1])
            if os.path.exists(final_model_dir):
                log.info(f"{model_name} already exists, skipping...")
                continue

            tokenizer = AutoTokenizer.from_pretrained(
                model_config["tokenizer"] if "tokenizer" in model_config else model_name,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",
            )
            if "llama" in model_name.lower():
                eot_token_id = 128009
                eos_token_id = 128001
                tokenizer.pad_token_id = 128004
                tokenizer.eos_token_id = eos_token_id
                tokenizer.add_eos_token = False
                eos_token = tokenizer.eos_token
            else:
                eos_token = tokenizer.eos_token
                bos_token = tokenizer.bos_token or ""
                special_tokens = {"pad_token": "[PAD]"}
                tokenizer.add_special_tokens(special_tokens)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.add_eos_token = False
            model.resize_token_embeddings(len(tokenizer))

            def preprocess_function(examples):
                trace_colname = 'rewrite_trace'
                suffix_len = -1 if "llama" in model_name.lower() else -2
                # Create chat format messages for each example
                messages = [[
                    {"role": "system", "content": cfg.instruction_generation},
                    {"role": "user", "content": problem.strip()},
                    {"role": "assistant", "content": response.strip()}]
                    for problem, response in zip(examples["problem"], examples[trace_colname])]
                tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                tokens = [toks[:suffix_len] for toks in tokens]
                return {"input_ids": tokens}

            train_dataset = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=16384,
                num_proc=96,
                remove_columns=list(dataset.column_names),
                desc="Preprocessing train dataset",
                load_from_cache_file=True
            )

            peft_parameters = LoraConfig(
                r=128,
                lora_alpha=128,
                lora_dropout=0.0,
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_parameters)
            if accelerator.is_main_process: model.print_trainable_parameters()

            num_gpus = accelerator.num_processes
            per_device_batch_size = 4
            gradient_accumulation_steps = 32 // (num_gpus * per_device_batch_size)
            epochs = 3
            learning_rate = float(6e-4)

            if accelerator.is_main_process:
                log.info(
                    f"hyperparams: bs={per_device_batch_size}, grad_accum={gradient_accumulation_steps}, "
                    f"effective_bs={per_device_batch_size * gradient_accumulation_steps * num_gpus}, "
                    f"epochs={epochs}, lr={learning_rate}"
                )

            training_args = SFTConfig(
                num_train_epochs=float(epochs),
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=float(learning_rate),
                weight_decay=float(0.01),
                warmup_ratio=float(0.03),
                lr_scheduler_type="cosine",
                max_grad_norm=float(1.0),
                remove_unused_columns=False,
                label_names=["labels"],
                completion_only_loss=True,

                bf16=True,
                fp16=False,
                
                seed=cfg.seed,
                log_level="info",
                logging_steps=10,
                logging_strategy="steps",

                save_strategy="no",
                report_to="none",  # disable wandb and others

                # Performance
                gradient_checkpointing=False,  # disabled for speed
                gradient_checkpointing_kwargs={"use_reentrant": False},
                group_by_length=True,
            )
            
            # Trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
            # Training
            trainer.train()

            if accelerator.is_main_process:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    return_dict=True,
                    device_map="cpu",
                )
                base_model.resize_token_embeddings(len(tokenizer))
                adapter_dir = os.path.join(final_model_dir, "adapter")
                trainer.save_model(adapter_dir)
                model_to_merge = PeftModel.from_pretrained(base_model, adapter_dir)
                merged_model = model_to_merge.merge_and_unload()
                merged_model.save_pretrained(final_model_dir)
                tokenizer.save_pretrained(final_model_dir)
                log.info(f"Finetuned {model_name}. Saved at {final_model_dir}")

            model, tokenizer = accelerator.free_memory(model, tokenizer)
            del model, tokenizer, train_dataset
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()



def run_scoring(cfg):
    accelerator = Accelerator()

    warmup_models(cfg, accelerator)

    instruction_path = os.path.join(cfg.working_dir, cfg.experiment_folder, "candidate_instructions.yaml")
    with open(instruction_path, 'r') as f:
        instruction_data = yaml.safe_load(f)
    candidates_list = instruction_data['candidate_instructions']
    instructions = []
    for item in candidates_list:
        if accelerator.is_main_process: log.info(f"Loading traces for instruction: {item['name']}")
        dataset_path = os.path.join(cfg.working_dir, item['name'])

        # Create an Instruction object and load its data
        instruction = Instruction(item['name'], item['instruction'])
        instruction.traces = datasets.load_from_disk(dataset_path)
        if cfg.rescore:
            instruction.score = None
        else:
            instruction.score = item.get('score', 0.0)
        instructions.append(instruction)

    if cfg.score_type == "loss":
        instructions_with_scores = score_instructions_loss(cfg, instructions, accelerator)
        if accelerator.is_main_process:
            log.info(f"Scoring done.")
            for instruction in instructions_with_scores:
                log.info(f"Instruction: {instruction.name}, Score: {instruction.score}")
            save_path = os.path.join(cfg.working_dir, cfg.experiment_folder, "instructions_and_scores.yaml")
            with open(save_path, "w") as f:
                f.write("candidate_instructions:\n")
                for instruction in instructions_with_scores:
                    f.write(f"  - name: {instruction.name}\n")
                    f.write(f"    instruction: |\n")
                    for line in instruction.text.splitlines():
                        f.write(f"      {line}\n")
                    f.write(f"    score: {instruction.score}\n")
    elif cfg.score_type == "acc":
        score_instructions_acc(cfg, instructions, accelerator)
    else:
        raise ValueError(f"Unknown score type: {cfg.score_type}")


if __name__ == "__main__":
    config_manager = ConfigManager()
    cfg = config_manager.get_config()

    run_scoring(cfg)