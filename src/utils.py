import argparse
import yaml
import logging
import os
import torch
from transformers import set_seed
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
)
MMLU_SYSTEM_PROMPT = (
    "Solve the given question step by step. You will finish your answer with \\boxed{{ANSWER}} where ANSWER is the correct letter choice.\n"
)
MMLU_ANSWER_FORCE_STRING = "\n\n**Final Choice**\n\\[\\boxed{"

def init(seed=42):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    cuda_capability = torch.cuda.get_device_capability()
    if cuda_capability[0] >= 8:  # Ampere or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_SDP_ATTENTION"] = "never"

class ConfigManager:
    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Generate and rewrite teacher reasoning traces.")
        parser.add_argument("--config", type=str, default="config/generate.yaml", help="Path to the configuration YAML file.")
        parser.add_argument("--seed", type=int, default=888, help="Random seed.")
        parser.add_argument("--working_dir", type=str, help="Directory for datasets and results.")
        parser.add_argument("--experiment_folder", type=str, help="Folder name for this experiment.")
        parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name (gsm8k, math, mmlu, mmlu-pro).")
        parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="Enable debug mode (small subset).")
        parser.add_argument("--num_samples", type=int, help="If set, restrict dataset to the first N samples.")
        return parser

    def get_config(self):
        args = self.parser.parse_args()

        # Load config from YAML file
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            log.warning(f"Config file not found at {args.config}. Using defaults and CLI args only.")
            config = {}

        # Override with CLI arguments if they are provided
        for key, value in vars(args).items():
            # Check for None is important because CLI flags with BooleanOptionalAction can be None
            if value is not None:
                config[key] = value
        
        # Convert to a namespace for dot notation access
        cfg = argparse.Namespace(**config)

        if "gsm8k" not in cfg.dataset and hasattr(cfg, "max_new_tokens"):
            cfg.max_new_tokens = 2048

        log.info(f"Configs:\n{yaml.dump(vars(cfg), sort_keys=False, default_flow_style=False)}")
        return cfg

def load_gsm8k(split="train"):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.rename_columns({"question": "problem", "answer": "solution"})
    dataset.shuffle(seed=42)
    train_size = int(len(dataset) * 0.7)
    if split == "train":
        return dataset.select(range(train_size))
    elif split == "holdout":
        return dataset.select(range(train_size, len(dataset)))
    elif split == "test":
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        dataset = dataset.rename_columns({"question": "problem", "answer": "solution"})
        return dataset
    else:
        raise ValueError("split must be either 'train', 'test', or 'holdout'")

def load_hendrycks_math_dataset(split="train"):

    if split not in ["train", "test", "holdout"]:
        raise ValueError("split must be either 'train', 'test', or 'holdout'")
    ds_split = "test" if split == "test" else "train"
    subsets = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    datasets = [load_dataset('EleutherAI/hendrycks_math', s, split=ds_split) for s in subsets]
    dataset = concatenate_datasets(datasets)

    if ds_split == "test":
        return dataset

    dataset = dataset.shuffle(seed=42)
    train_size = int(len(dataset) * 0.7)
    if split == "train":
        dataset = dataset.select(range(train_size))
    elif split == "holdout":
        dataset = dataset.select(range(train_size, len(dataset)))
    return dataset

def load_mmlu_pro(split="train", category="all", train_ratio=0.7, seed=42):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    ds = dataset['test']
    
    # Get unique categories
    unique_categories = set(ds['category'])
    
    # Determine which categories to include
    if category == "all":
        target_categories = unique_categories
    elif isinstance(category, str):
        if category not in unique_categories:
            raise ValueError(f"Category '{category}' not found. Available: {sorted(unique_categories)}")
        target_categories = {category}
    elif isinstance(category, (list, tuple)):
        target_categories = set(category)
        missing = target_categories - unique_categories
        if missing:
            raise ValueError(f"Categories not found: {sorted(missing)}. Available: {sorted(unique_categories)}")
    else:
        raise ValueError("category must be 'all', a string, or a list of strings")
    
    # Stratified sampling: split each category proportionally
    train_indices = []
    test_indices = []
    
    for cat in sorted(list(target_categories)):
        # Get indices for this category
        cat_indices = [i for i, c in enumerate(ds['category']) if c == cat]
        
        # Shuffle indices for this category
        import random
        random.seed(seed)
        random.shuffle(cat_indices)
        
        # Split this category
        cat_train_size = int(len(cat_indices) * train_ratio)
        cat_train_indices = cat_indices[:cat_train_size]
        cat_test_indices = cat_indices[cat_train_size:]
        
        train_indices.extend(cat_train_indices)
        test_indices.extend(cat_test_indices)
    
    # Select the appropriate split
    if split == "train":
        ret = ds.select(train_indices)
    elif split == "test":
        ret = ds.select(test_indices)
    else:
        raise ValueError("split must be 'train' or 'test'")
    
    def preprocess(ex):
        def filter_none(options):
            return [opt for opt in options if opt is not None]
        
        choices = filter_none(ex['options'])
        prompt = f"{ex['question']}\n"
        prompt += '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        sol = f"\\boxed{{{ex['answer']}}}"
        return {'problem': prompt, 'solution': sol, 'category': ex['category']}
    
    return ret.map(preprocess, remove_columns=ret.column_names)

def load_mmlu(split="train"):
    if split in ["train", "holdout"]:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    else:
        ds = load_dataset("cais/mmlu", "all", split="test")
    ds.shuffle(seed=42)
    train_size = int(len(ds) * 0.7)
    if split == "train":
        ret = ds.select(range(train_size))
    elif split == "holdout":
        ret = ds.select(range(train_size, len(ds)))
    elif split == "test":
        ret = ds
    else:
        raise ValueError("split must be train, test, or holdout")

    unique_categories = set(ret['subject'])
    log.info(f"dataset has categories: {unique_categories}")
    log.info(f"first sample from the dataset:\n{ret[0]}")
    
    def to_math_format(mmlu_ds):
        def format_example(ex):
            choices = ex['choices']
            prompt = f"{ex['question']}\n"
            prompt += '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return prompt

        def transform(ex):
            problem = format_example(ex)
            letter = chr(65 + ex['answer'])
            sol = "\\boxed{" + letter + "}"
            return {'problem': problem, 'solution': sol, 'category': ex['subject']}

        return mmlu_ds.map(transform, remove_columns=mmlu_ds.column_names)

    return to_math_format(ret)

mmlu_subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

mmlu_categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def load_mmlu_for_su(split="train", category="all"):
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds.shuffle(seed=42)
    train_ratio = 0.7

    unique_categories = set(ds['subject'])
    if category == "all":
        target_categories = unique_categories
    elif isinstance(category, str):
        if category not in unique_categories:
            raise ValueError(f"Category '{category}' not found. Available: {sorted(unique_categories)}")
        target_categories = {category}
    elif isinstance(category, (list, tuple)):
        target_categories = set(category)
        missing = target_categories - unique_categories
        if missing:
            raise ValueError(f"Categories not found: {sorted(missing)}. Available: {sorted(unique_categories)}")
    else:
        raise ValueError("category must be 'all', a string, or a list of strings")
    
    # Stratified sampling: split each category proportionally
    train_indices = []
    test_indices = []
    
    for cat in sorted(list(target_categories)):
        # Get indices for this category
        cat_indices = [i for i, c in enumerate(ds['subject']) if c == cat]
        
        # Split this category
        cat_train_size = int(len(cat_indices) * train_ratio)
        cat_train_indices = cat_indices[:cat_train_size]
        cat_test_indices = cat_indices[cat_train_size:]
        
        train_indices.extend(cat_train_indices)
        test_indices.extend(cat_test_indices)
    
    # Select the appropriate split
    if split == "train":
        ret = ds.select(train_indices)
    elif split == "test":
        ret = ds.select(test_indices)
    else:
        raise ValueError("split must be 'train' or 'test'")
    
    def to_math_format(mmlu_ds):
        def format_example(ex):
            choices = ex['choices']
            prompt = f"{ex['question']}\n"
            prompt += '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return prompt

        def format_cat(cat):
            subcats = mmlu_subcategories.get(cat, ["other"])
            for main_cat, subcat_list in mmlu_categories.items():
                if any(sub in subcats for sub in subcat_list):
                    return main_cat
            return "other"

        def transform(ex):
            problem = format_example(ex)
            letter = chr(65 + ex['answer'])
            sol = "\\boxed{" + letter + "}"
            cat = format_cat(ex['subject'])
            return {'problem': problem, 'solution': sol, 'category': cat}

        return mmlu_ds.map(transform, remove_columns=mmlu_ds.column_names)

    return to_math_format(ret)