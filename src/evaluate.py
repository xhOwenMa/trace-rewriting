"""Evaluate a student (or teacher) model on a math/MMLU benchmark.

For every example we compute two accuracies:
  - raw: verify the model's free-form trace against the ground truth.
  - answer-forced (AF): append a final-answer prompt to the trace, generate a short
    continuation, and verify it. This absorbs formatting noise that often hides a correct
    answer inside a long reasoning trace.
"""

import argparse
import os
import sys

import yaml
from loguru import logger
from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from utils import (
    ANSWER_FORCE_STRING,
    MMLU_ANSWER_FORCE_STRING,
    MMLU_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    init,
    load_gsm8k,
    load_hendrycks_math_dataset,
    load_mmlu_for_su,
    load_mmlu_pro,
)

logger.remove()
logger.add(sys.stdout, level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on math / MMLU benchmarks.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, help="Comma-separated list of datasets (gsm8k, math, mmlu, mmlu-pro).")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--save_eval_dataset", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None, help="If set, restrict test set to the first N samples.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def log_color(content, title=""):
    Console().print(Panel(Text(str(content)), title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def is_correct(example, trace_colname):
    """Verify a trace against the ground-truth solution, tolerating answer-forcing splits."""
    trace = example[trace_colname]
    try:
        soln = parse(example["solution"])
        extract_cfg = [ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig()]
        for force_str in (ANSWER_FORCE_STRING, MMLU_ANSWER_FORCE_STRING):
            if force_str in trace:
                parts = trace.split(force_str)
                candidates = [trace, force_str.join(parts[:-1]), parts[-1]]
                return {"is_correct": any(verify(soln, parse(c, extraction_config=extract_cfg)) for c in candidates)}
        return {"is_correct": verify(soln, parse(trace, extraction_config=extract_cfg))}
    except Exception:
        logger.warning("Failed to parse/verify — defaulting to incorrect.")
        return {"is_correct": False}


def mmlu_acc_by_subject(final_dataset):
    df = final_dataset.to_pandas()
    breakdown = {}
    for subject, group in df.groupby("category"):
        raw = float(group["is_raw_correct"].mean())
        af = float(group["is_af_correct"].mean())
        breakdown[subject] = {"raw_accuracy": raw, "af_accuracy": af, "num_examples": len(group)}
        logger.info(f"Subject: {subject}, Raw: {raw:.4f}, AF: {af:.4f}")
    return breakdown


def load_test_split(dataset_name):
    if "gsm8k" in dataset_name:
        return load_gsm8k(split="test"), 1024
    if "math" in dataset_name:
        return load_hendrycks_math_dataset(split="test"), 2048
    if dataset_name == "mmlu":
        return load_mmlu_for_su("test"), 2048
    if dataset_name == "mmlu-pro":
        return load_mmlu_pro("test", ["math", "physics", "philosophy", "psychology"]), 2048
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_chat_prompts(dataset, dataset_name, tokenizer):
    sys_prompt = MMLU_SYSTEM_PROMPT if "mmlu" in dataset_name else SYSTEM_PROMPT

    def preprocess(examples):
        messages = [[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": problem.strip() + "\n"},
        ] for problem in examples["problem"]]
        texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"input_ids": texts, "seq_lengths": [len(t) for t in texts]}

    return dataset.map(preprocess, batched=True, num_proc=96, desc="Preprocessing", load_from_cache_file=True)


def main():
    args = parse_args()
    init(args.seed)

    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    sampling_params_af = SamplingParams(temperature=0, top_p=0.95, max_tokens=32)

    for dataset_name in args.dataset.split(","):
        save_path = os.path.join(args.out_dir, dataset_name)
        os.makedirs(args.out_dir, exist_ok=True)
        dataset, max_new_tokens = load_test_split(dataset_name)
        sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=max_new_tokens)

        if args.debug:
            dataset = dataset.select(range(torch.cuda.device_count() * 2 * 64))
        if args.num_samples is not None:
            dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        logger.info(f"Evaluating on {dataset_name} ({len(dataset)} examples)")

        proc_dataset = build_chat_prompts(dataset, dataset_name, tokenizer)
        eg_solution = dataset[0]["solution"]
        log_color(
            f"### Problem:\n{dataset[0]['problem']}\n### Solution:\n{eg_solution}\n"
            f"### Parsed ground truth: {parse(eg_solution)}",
            title="First example",
        )
        log_color(proc_dataset[0]["input_ids"], title="Example input")

        prompts = list(proc_dataset["input_ids"])
        raw_outputs = model.generate(prompts, sampling_params)
        raw_traces = [o.outputs[0].text for o in raw_outputs]
        dataset = dataset.add_column("trace", raw_traces)
        dataset = dataset.map(is_correct, fn_kwargs={"trace_colname": "trace"}, desc="Scoring raw").rename_column("is_correct", "is_raw_correct")

        force_str = MMLU_ANSWER_FORCE_STRING if "mmlu" in dataset_name else ANSWER_FORCE_STRING
        af_inputs = [trace + force_str for trace in raw_traces]
        log_color(af_inputs[0], title="Example input after answer-forcing")
        af_outputs = model.generate(af_inputs, sampling_params_af)
        af_traces = [prefix + o.outputs[0].text for prefix, o in zip(af_inputs, af_outputs)]
        dataset = dataset.add_column("trace_af", af_traces)
        dataset = dataset.map(is_correct, fn_kwargs={"trace_colname": "trace_af"}, desc="Scoring AF").rename_column("is_correct", "is_af_correct")
        torch.cuda.empty_cache()

        df = dataset.to_pandas()
        trace_len_stats = {k: float(v) for k, v in df["trace"].map(lambda x: len(tokenizer.encode(x))).describe().items()}
        raw_accuracy = float(df["is_raw_correct"].mean())
        af_accuracy = float(df["is_af_correct"].mean())
        mmlu_subject_acc = mmlu_acc_by_subject(dataset) if "mmlu" in dataset_name else None

        log_color(
            f"Dataset: {dataset_name}, Raw Accuracy: {raw_accuracy:.4f}, AF Accuracy: {af_accuracy:.4f}\n"
            f"Trace length stats: {trace_len_stats}",
            title="Evaluation Results",
        )

        results = {
            "seed": args.seed,
            "model_name_or_path": args.model_name_or_path,
            "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
            "trace_save_path": save_path,
            "dataset": f"{dataset_name}_test",
            "temperature": args.temperature,
            "raw_accuracy": raw_accuracy,
            "af_accuracy": af_accuracy,
            "mmlu_subject_accuracies": mmlu_subject_acc,
            "trace_len_stats": trace_len_stats,
        }
        with open(save_path + ".yaml", "w") as f:
            yaml.dump(results, f, sort_keys=False)
        df.to_json(save_path + ".jsonl", orient="records", lines=True)
        if args.save_eval_dataset:
            dataset.save_to_disk(save_path)
            logger.info(f"Saved evaluated dataset to {save_path}")


if __name__ == "__main__":
    main()
