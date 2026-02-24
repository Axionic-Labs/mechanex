import os
import json
import re
from typing import List, Dict, Optional, Any
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import torch
from ..interfaces import BaseEnvironment
from ..ara.module import ARAModule, ARAConfig, TeacherProvider

# --- REWARD FUNCTION LOGIC (PRESERVED AS IS) ---

class RewardScore:
    def __init__(self, total, format_score, constraint_score, type_score, accuracy_score=1.0, safety_score=1.0):
        self.total = total
        self.format_score = format_score
        self.constraint_score = constraint_score
        self.type_score = type_score
        self.accuracy_score = accuracy_score
        self.safety_score = safety_score

def unwrap_text(x):
    if isinstance(x, list) and len(x) > 0:
        if isinstance(x[0], dict) and "content" in x[0]:
            return x[0]["content"]
        if isinstance(x[0], str):
            return x[0]
    if isinstance(x, str):
        return x
    return ""

def pred_tool_name(response):
    text = unwrap_text(response)
    pattern = r'(?s)<tool_call>.*?"name"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, text)
    return (matches[0], {}) if matches else (None, {})

def extract_tool_call_gt(response):
    if isinstance(response, list):
        if len(response) == 0:
            return None, {}
        response = response[0]
    if not isinstance(response, str):
        return None, {}
    match = re.search(r'{"name":\s*"([^"]+)"', response)
    return (match.group(1), {}) if match else (None, {})

class RLTrainerModule:
    def __init__(
        self,
        model_name: str,
        tool_schemas: List[Dict[str, Any]],
        ara_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "rl_output"
    ):
        self.model_name = model_name
        self.tool_schemas = tool_schemas
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize ARA
        ara_cfg = ARAConfig(**(ara_config or {
            "teacher_provider": TeacherProvider.GOOGLE,
            "teacher_model": "gemini-2.0-flash"
        }))
        self.ara = ARAModule(config=ara_cfg)
        
        self.tool_names = [x["name"] for x in self.tool_schemas]
        self.reverse_tool_map = {tool: i for i, tool in enumerate(self.tool_names)}
        self.ara_reward_cache = {}

    def make_ara_reward_fn(self, schema):
        original_reward_fn = self.ara.compile(schema)
        def adapted_reward_fn(prompt, response):
            res_dict = original_reward_fn(response, prompt)
            bs = res_dict.get("breakdown", {})
            return RewardScore(
                total=res_dict.get("score", 0.0),
                format_score=bs.get("format", 0.0),
                constraint_score=bs.get("grounding", 0.0),
                type_score=bs.get("type_valid", 0.0),
                accuracy_score=bs.get("reasoning", 0.0),
                safety_score=1.0,
            )
        return adapted_reward_fn

    def get_cached_reward_fn(self, tool_name):
        if tool_name not in self.ara_reward_cache:
            schema_idx = self.reverse_tool_map[tool_name]
            schema = self.tool_schemas[schema_idx]
            self.ara_reward_cache[tool_name] = self.make_ara_reward_fn(schema)
        return self.ara_reward_cache[tool_name]

    def grpo_reward_function(self, prompts, completions, ground_truth, **kwargs):
        rewards = []
        large_negative_reward = kwargs.get("large_negative_reward", -10.0)
        
        for prompt, pred_resp, gt_resp in zip(prompts, completions, ground_truth):
            pred_tool, _ = pred_tool_name(pred_resp)
            gt_tool, _ = extract_tool_call_gt(gt_resp)

            if pred_tool is None or pred_tool != gt_tool:
                rewards.append(large_negative_reward)
                continue

            try:
                reward_fn = self.get_cached_reward_fn(pred_tool)
                prompt_text = unwrap_text(prompt)
                response_text = unwrap_text(pred_resp)
                score = reward_fn(prompt_text, response_text)
                rewards.append(score.total)
            except Exception as e:
                # Failure in reward function (e.g. parsing) defaults to 0 reward
                # This matches train_grpo.py behavior
                # print(f"Warning: Reward function failed: {e}")
                rewards.append(0.0)
        return rewards

    def format_dataset(self, dataset: Dataset) -> Dataset:
        tool_definitions = ", ".join(self.tool_names)
        
        def format_example(example):
            instruction = f"""You are a reasoning agent. You must output a complete tool call in the exact format shown below.

Format your response EXACTLY as:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Available Tools: {tool_definitions}

Task: {example['prompt']}

CRITICAL REQUIREMENTS:
1. You MUST include the complete JSON with both "name" AND "arguments"
2. The "arguments" field MUST contain all required parameters as a dictionary
3. Do NOT output only the tool name
4. Do NOT include any text before or after the <tool_call> tags
5. Do NOT include explanations, reasoning, or markdown

Your output must be ONLY the complete <tool_call> block with arguments included."""

            # Determine ground truth
            if 'response' in example:
                ground_truth = example['response']
            elif 'expected_entities' in example and 'tool_name' in example:
                # Construct logical ground truth from seed data
                # This allows training directly on seed prompts without pre-generated trajectories
                gt_call = {
                    "name": example['tool_name'],
                    "arguments": example['expected_entities']
                }
                ground_truth = json.dumps(gt_call)
            else:
                # Fallback or error - for now allow empty but reward will likely be 0
                ground_truth = ""

            return {
                "prompt": [{"role": "user", "content": instruction}],
                "ground_truth": ground_truth
            }

        return dataset.map(format_example)

    def train(self, dataset: Dataset, train_args_patch: Optional[Dict[str, Any]] = None):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        default_args = {
            "learning_rate": 5e-6,
            "num_train_epochs": 1,
            "logging_steps": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "output_dir": self.output_dir,
            "report_to": "none",
            "num_generations": 4,
            "max_prompt_length": 512,
            "max_completion_length": 256,
        }
        if train_args_patch:
            default_args.update(train_args_patch)
            
        training_args = GRPOConfig(**default_args)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[self.grpo_reward_function],
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final_rl_model"))
        return os.path.join(self.output_dir, "final_rl_model")
