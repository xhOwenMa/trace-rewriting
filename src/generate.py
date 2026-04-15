"""Generate teacher reasoning traces and rewritten versions via an OpenAI-compatible API.

Pipeline (all steps idempotent — existing columns are preserved):
  1. Generate original traces from problems with `teacher_model_name`
     (skipped if `original_traces_path` set).
  2. Rewrite originals with `rewriter_model_name` using `rewrite_instruction`
     (skipped if dataset already has `rewrite_trace`).
  3. Answer-forcing accuracy: feed (problem + trace) back to the rewriter and ask for
     the final answer only; compare against ground truth. Skippable via `--skip_af`.

Point `OPENAI_BASE_URL` at any OpenAI-compatible endpoint (e.g. `vllm serve` on localhost).
If teacher ≠ rewriter, run in two stages: first with `--skip_rewrite --skip_af` against
the teacher server, then re-run with `--original_traces_path <...>` against the rewriter.
"""

import argparse
import asyncio
import difflib
import logging
import os
import shutil
import yaml

import datasets
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from utils import (
    ANSWER_FORCE_STRING,
    MMLU_ANSWER_FORCE_STRING,
    MMLU_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ConfigManager,
    init,
    load_gsm8k,
    load_hendrycks_math_dataset,
    load_mmlu_for_su,
    load_mmlu_pro,
)
from evaluate import is_correct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CONCURRENCY_LIMIT = 128

client = AsyncOpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
)


def load_train_split(dataset_name):
    if "gsm8k" in dataset_name:
        return load_gsm8k(split="train")
    if "math" in dataset_name:
        return load_hendrycks_math_dataset(split="train")
    if dataset_name == "mmlu":
        return load_mmlu_for_su("train")
    if dataset_name == "mmlu-pro":
        return load_mmlu_pro("train", ["math", "physics", "philosophy", "psychology"])
    raise ValueError(f"Unsupported dataset: {dataset_name}")


async def call_once(messages, *, model, max_tokens, temperature, reasoning_effort=None):
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    while True:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            choice = response.choices[0]
            content = getattr(choice.message, "content", None) or getattr(choice.message, "reasoning_content", None)
            if content and content.strip():
                return content
            log.warning("Empty model output — retrying.")
        except Exception as e:
            log.error(f"Request failed: {e}")
        await asyncio.sleep(1)


async def batch_call(build_messages, user_inputs, *, model, max_tokens, temperature, reasoning_effort=None, desc=""):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def _one(item):
        async with semaphore:
            return await call_once(
                build_messages(item),
                model=model, max_tokens=max_tokens, temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

    return await tqdm.gather(*[_one(x) for x in user_inputs], desc=desc)


def drop_empty(ds, col):
    keep = [i for i, t in enumerate(ds[col]) if t and t.strip()]
    if len(keep) < len(ds):
        log.warning(f"Dropping {len(ds) - len(keep)} examples with empty {col}")
    return ds.select(keep)


def score_column(ds, trace_col, out_col):
    ds = ds.map(is_correct, fn_kwargs={"trace_colname": trace_col}, desc=f"Scoring {trace_col}")
    return ds.rename_column("is_correct", out_col)


async def generate_originals(cfg, ds):
    log.info(f"Generating {len(ds)} original traces with {cfg.teacher_model_name}")
    def messages_for(problem):
        return [
            {"role": "system", "content": cfg.generation_instruction},
            {"role": "user", "content": problem},
        ]
    traces = await batch_call(
        messages_for, [x["problem"] for x in ds],
        model=cfg.teacher_model_name,
        max_tokens=cfg.max_new_tokens, temperature=cfg.temperature, desc="Original traces",
    )
    ds = ds.add_column("original_trace", traces)
    ds = drop_empty(ds, "original_trace")
    return score_column(ds, "original_trace", "is_original_correct")


async def rewrite_traces(cfg, ds):
    log.info(f"Rewriting {len(ds)} traces with {cfg.rewriter_model_name}")
    def messages_for(trace):
        return [
            {"role": "system", "content": cfg.rewrite_instruction},
            {"role": "user", "content": trace},
        ]
    rewrites = await batch_call(
        messages_for, [x["original_trace"] for x in ds],
        model=cfg.rewriter_model_name,
        max_tokens=cfg.max_new_tokens, temperature=cfg.temperature, desc="Rewritten traces",
    )
    ds = ds.add_column("rewrite_trace", rewrites)
    ds = drop_empty(ds, "rewrite_trace")
    return score_column(ds, "rewrite_trace", "is_rewritten_correct")


async def answer_forcing(cfg, ds):
    """Feed (problem + trace) back through the rewriter with a strict final-answer prompt."""
    is_mmlu = "mmlu" in cfg.dataset
    system_text = MMLU_SYSTEM_PROMPT if is_mmlu else SYSTEM_PROMPT
    tag = "Only output the **Final Choice:**" if is_mmlu else "Only output the **Final Answer:**"
    effort = getattr(cfg, "af_reasoning_effort", None)

    def build_prompts(trace_col):
        return [f"{system_text}{x['problem']}\n\n{x[trace_col]}\n\n{tag}\n" for x in ds]

    def messages_for(user_text):
        return [{"role": "user", "content": user_text}]

    log.info(f"Answer-forcing on original_trace with {cfg.rewriter_model_name}")
    orig_af = await batch_call(
        messages_for, build_prompts("original_trace"),
        model=cfg.rewriter_model_name,
        max_tokens=1024, temperature=0.0, reasoning_effort=effort, desc="AF original",
    )
    ds = ds.add_column("original_trace_af_output", orig_af)
    ds = score_column(ds, "original_trace_af_output", "is_original_af_correct")

    if "rewrite_trace" in ds.column_names:
        log.info(f"Answer-forcing on rewrite_trace with {cfg.rewriter_model_name}")
        rew_af = await batch_call(
            messages_for, build_prompts("rewrite_trace"),
            model=cfg.rewriter_model_name,
            max_tokens=1024, temperature=0.0, reasoning_effort=effort, desc="AF rewrite",
        )
        ds = ds.add_column("rewrite_trace_af_output", rew_af)
        ds = score_column(ds, "rewrite_trace_af_output", "is_rewrite_af_correct")
    return ds


def record_accuracies(cfg, ds):
    df = ds.to_pandas()
    for col, attr in (
        ("is_original_correct", "original_accuracy"),
        ("is_rewritten_correct", "rewritten_accuracy"),
        ("is_original_af_correct", "original_af_accuracy"),
        ("is_rewrite_af_correct", "rewrite_af_accuracy"),
    ):
        if col in df.columns:
            setattr(cfg, attr, float(df[col].mean()))
    summary = ", ".join(f"{k}={v:.4f}" for k, v in vars(cfg).items() if k.endswith("_accuracy"))
    log.info(f"Accuracies: {summary}")

    if "mmlu" in cfg.dataset and "category" in df.columns:
        breakdown = {}
        acc_cols = [c for c in
                    ("is_original_correct", "is_rewritten_correct",
                     "is_original_af_correct", "is_rewrite_af_correct") if c in df.columns]
        for subject, group in df.groupby("category"):
            breakdown[subject] = {c: float(group[c].mean()) for c in acc_cols}
            breakdown[subject]["num_examples"] = len(group)
            log.info(f"  {subject}: {breakdown[subject]}")
        cfg.subject_accuracies = breakdown


def write_sample_report(ds, path, num_samples=5):
    n = min(num_samples, len(ds))
    parts = [f"# Rewriting Results Report\n\n**Total samples:** {len(ds)}\n\n"]
    for i, ex in enumerate(ds.select(range(n))):
        diff = "".join(difflib.unified_diff(
            ex["original_trace"].splitlines(keepends=True),
            ex["rewrite_trace"].splitlines(keepends=True),
            fromfile="original", tofile="rewritten",
        ))
        parts.append(
            f"## Sample {i + 1}\n\n"
            f"### Problem\n\n```\n{ex['problem']}\n```\n\n"
            f"### Original Trace\n\n```\n{ex['original_trace']}\n```\n\n"
            f"### Rewritten Trace\n\n```\n{ex['rewrite_trace']}\n```\n\n"
            f"### Diff\n\n```diff\n{diff}\n```\n\n---\n\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def output_paths(cfg):
    base = "debug_output" if cfg.debug else f"{cfg.working_dir}/{cfg.dataset}/{cfg.experiment_folder}"
    os.makedirs(base, exist_ok=True)
    return {
        "dataset": f"{base}/traces",
        "report": f"{base}/report.md",
        "config": f"{base}/config.yaml",
    }


async def run(cfg):
    if cfg.original_traces_path:
        log.info(f"Loading pre-generated originals from {cfg.original_traces_path}")
        existing = datasets.load_from_disk(cfg.original_traces_path)
        ds = load_train_split(cfg.dataset)
        if len(existing) != len(ds):
            raise ValueError(f"Row count mismatch: {len(existing)} traces vs {len(ds)} dataset rows")
        ds = ds.add_column("original_trace", existing["original_trace"])
        if "is_original_correct" in existing.column_names:
            ds = ds.add_column("is_original_correct", existing["is_original_correct"])
        else:
            ds = score_column(ds, "original_trace", "is_original_correct")
        if "rewrite_trace" in existing.column_names:
            ds = ds.add_column("rewrite_trace", existing["rewrite_trace"])
            if "is_rewritten_correct" in existing.column_names:
                ds = ds.add_column("is_rewritten_correct", existing["is_rewritten_correct"])
            else:
                ds = score_column(ds, "rewrite_trace", "is_rewritten_correct")
        if cfg.debug:
            ds = ds.select(range(8))
        if getattr(cfg, "num_samples", None):
            ds = ds.select(range(min(cfg.num_samples, len(ds))))
    else:
        ds = load_train_split(cfg.dataset)
        if cfg.debug:
            ds = ds.select(range(8))
        if getattr(cfg, "num_samples", None):
            ds = ds.select(range(min(cfg.num_samples, len(ds))))
        ds = await generate_originals(cfg, ds)

    if "rewrite_trace" not in ds.column_names and not cfg.skip_rewrite:
        ds = await rewrite_traces(cfg, ds)

    if not cfg.skip_af:
        ds = await answer_forcing(cfg, ds)

    record_accuracies(cfg, ds)

    paths = output_paths(cfg)
    if os.path.exists(paths["dataset"]):
        shutil.rmtree(paths["dataset"])
    ds.save_to_disk(paths["dataset"])
    if "rewrite_trace" in ds.column_names:
        write_sample_report(ds, paths["report"])
    with open(paths["config"], "w") as f:
        yaml.dump(vars(cfg), f, sort_keys=False)
    log.info(f"Saved dataset to {paths['dataset']}")


if __name__ == "__main__":
    mgr = ConfigManager()
    mgr.parser.add_argument("--skip_rewrite", action="store_true", help="Skip the rewrite step.")
    mgr.parser.add_argument("--skip_af", action="store_true", help="Skip answer-forcing evaluation.")
    cfg = mgr.get_config()
    init(cfg.seed)
    asyncio.run(run(cfg))
