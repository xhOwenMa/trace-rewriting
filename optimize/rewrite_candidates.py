"""Stage 1 of prompt optimization: rewrite traces with each candidate instruction.

Reads `candidate_instructions.yaml` from the experiment directory, and for every
candidate, generates rewritten traces with the teacher model and saves them to disk.
"""

import logging
import os
import warnings

import datasets
import torch
import yaml
from vllm import LLM, SamplingParams

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


def generate_and_save_rewrites(cfg):
    os.makedirs(cfg.working_dir, exist_ok=True)
    model = LLM(model=cfg.rewriter_model_name, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    dataset = datasets.load_from_disk(cfg.original_traces_path)
    dataset = dataset.shuffle(seed=137).select(range(cfg.dataset_size_used_for_optimize))
    dataset = dataset.select_columns(["problem", "solution", "original_trace"])

    instruction_path = os.path.join(cfg.working_dir, cfg.experiment_folder, "candidate_instructions.yaml")
    with open(instruction_path, "r") as f:
        instruction_data = yaml.safe_load(f)
    instructions = [Instruction(item["name"], item["instruction"]) for item in instruction_data["candidate_instructions"]]
    log.info(f"Loaded {len(instructions)} candidate instructions: {[i.name for i in instructions]}")

    for instruction in instructions:
        save_path = os.path.join(cfg.working_dir, instruction.name)
        if os.path.exists(save_path):
            log.info(f"Skipping existing traces for {instruction.name}")
            continue

        log.info(f"Generating traces for {instruction.name}")
        prompts = [
            cfg.rewrite_prompt_template.format(instruction=instruction.text, original_trace=sample["original_trace"])
            for sample in dataset
        ]
        outputs = model.generate(prompts, sampling_params)
        rewrites = [o.outputs[0].text for o in outputs]
        dataset.add_column("rewrite_trace", rewrites).save_to_disk(save_path)
        log.info(f"Saved to {save_path}")


if __name__ == "__main__":
    cfg = ConfigManager().get_config()
    generate_and_save_rewrites(cfg)
