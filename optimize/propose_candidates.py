import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from openai import OpenAI
import os
import yaml
import re

import warnings
warnings.filterwarnings("ignore")

from utils import ConfigManager


def optimize_instructions(cfg, prompt):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY to use the optimizer LLM (OpenRouter).")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    model_name = os.environ.get("OPTIMIZER_MODEL", "deepseek/deepseek-chat-v3.1:free")

    all_completions = []
    while len(set(all_completions)) < 3:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
            all_completions.append(completion.choices[0].message.content)
        except Exception as e:
            log.error(f"Error during optimization: {e}")
            continue
    return all_completions


def create_new_instruction_data(cfg, old_instructions, new_instructions):
    formatted_new_instructions = []
    for i, instruction_text in enumerate(new_instructions):
        formatted_new_instructions.append({
            "name": f"{cfg.experiment_folder}_new_instruction_{i}",
            "instruction": instruction_text,
            "score": None  # Use None for unscored instructions
        })
    log.info(f"✅ Formatted {len(formatted_new_instructions)} new instructions.")

    try:
        # Find the 'generation' instruction dictionary
        generation_instruction = next(item for item in old_instructions if item['name'] == 'generation')
        generation_score = generation_instruction['score']
        log.info(f"🔍 Found 'generation' instruction with score: {generation_score}")
    except StopIteration:
        log.info("⚠️ Warning: 'generation' instruction not found. No old instructions will be kept.")
        generation_score = float('-inf')
    filtered_old = [
        item for item in old_instructions if item.get('score') is not None and item['score'] > generation_score
    ]
    log.info(f"✅ Filtered old instructions. Kept {len(filtered_old)} instructions with a score higher than {generation_score}.")

    final_candidate_list = filtered_old + formatted_new_instructions

    current_num = cfg.experiment_folder.split('_')[1]
    next_iter_folder = re.sub(r'iter_(\d+)', lambda m: f"iter_{int(m.group(1)) + 1}", cfg.experiment_folder)
    output_path = os.path.join(cfg.working_dir, next_iter_folder)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "candidate_instructions.yaml"), "w") as f:
    # with open("debug_output/candidate_instructions.yaml", "w") as f:
        f.write("candidate_instructions:\n")
        for instruction in final_candidate_list:
            f.write(f"  - name: {instruction['name']}\n")
            f.write(f"    instruction: |\n")
            # Ensure the instruction text itself is properly indented
            for line in instruction['instruction'].splitlines():
                f.write(f"      {line}\n")
            
            # Handle writing the score, including the None case
            score_value = instruction['score']
            if score_value is None:
                # This creates a 'null' value in the YAML file, which is standard
                f.write(f"    score:\n")
            else:
                f.write(f"    score: {score_value}\n")
    
    log.info(f"🚀 Successfully saved {len(final_candidate_list)} combined instructions to:\n{output_path}")


def format_prompt(cfg):
    prompt = cfg.optimization_prompt

    saved_path = os.path.join(cfg.working_dir, cfg.experiment_folder, "instructions_and_scores.yaml")
    with open(saved_path, 'r') as f:
        instruction_data = yaml.safe_load(f)

    candidates_list = instruction_data['candidate_instructions']
    sorted_instructions = sorted(candidates_list, key=lambda item: item['score'])
    # only use the top 10 instructions
    if len(candidates_list) > 10: sorted_instructions = sorted_instructions[-10:]

    for idx, item in enumerate(sorted_instructions):
        prompt += f"\n\nInstruction:\n{item['instruction']}Score: {item['score']}"

    prompt += "\n\nBased on the above examples, write a new instruction that is different from the old ones and has a score as high as possible. Only provide the new instruction. Do not add any conversational text."

    log.info(f"Prompt to be given to the Optimizor LLM:\n{prompt}")

    return prompt, sorted_instructions

if __name__ == "__main__":
    config_manager = ConfigManager()
    cfg = config_manager.get_config()

    prompt, sorted_old_instructions = format_prompt(cfg)
    new_instructions_from_optimization = optimize_instructions(cfg, prompt)

    create_new_instruction_data(cfg, sorted_old_instructions, new_instructions_from_optimization)