import os
import json
import re
import shutil
import glob
import logging
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainerCallback, 
    TrainerState, 
    TrainerState, 
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig
from ..interfaces import BaseEnvironment, BaseTeacher

logger = logging.getLogger(__name__)

class TopKCheckpointerCallback(TrainerCallback):
    """Callback to keep only the top K checkpoints based on a metric."""
    def __init__(self, top_k=10, metric="eval_loss", greater_is_better=False):
        self.top_k = top_k
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_checkpoints = [] # List of (metric_value, checkpoint_path)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get(self.metric)
        if current_metric is None:
            return

        # Checkpoints are usually saved at the same time or after evaluation
        # We look for the most recent checkpoint folder
        checkpoint_dir = args.output_dir
        checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not checkpoints:
            return

        last_checkpoint = checkpoints[0]
        
        # Add to history if not already there
        if not any(cp == last_checkpoint for _, cp in self.best_checkpoints):
            self.best_checkpoints.append((current_metric, last_checkpoint))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=self.greater_is_better)

        # Prune if over limit
        while len(self.best_checkpoints) > self.top_k:
            metric_val, path = self.best_checkpoints.pop(-1)
            if os.path.exists(path):
                logger.info(f"Pruning checkpoint {path} with {self.metric}={metric_val}")
                shutil.rmtree(path)

class SFTTrainerModule:
    def __init__(
        self,
        model_name: str,
        env: BaseEnvironment,
        teacher: BaseTeacher,
        tool_schemas: List[Dict[str, Any]],
        output_dir: str = "sft_output"
    ):
        self.model_name = model_name
        self.env = env
        self.teacher = teacher
        self.tool_schemas = tool_schemas
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def generate_training_data(self, prompts: List[str], samples: int = 2) -> List[Dict[str, str]]:
        data = []
        for p in tqdm(prompts, desc="Generating SFT data"):
            for _ in range(samples):
                trace = await self.teacher.generate_trace(p)
                if trace["success"] and trace["tool_call"]:
                    try:
                        # Even if tool execution fails, we might keep the traj if teacher is good,
                        # but original code skipped failed tool calls.
                        # We follow original logic.
                        result = self.env.execute_tool(trace["tool_call"])
                        if result.success:
                            data.append({"prompt": p, "response": trace["raw"]})
                    except Exception:
                        continue
        return data

    def format_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        tool_names = ", ".join([s["name"] for s in self.tool_schemas])
        
        def format_example(example):
            instruction = f"""You are a reasoning agent. You must output a complete tool call in the exact format shown below.

Format your response EXACTLY as:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Available Tools: {tool_names}

Task: {example['prompt']}

CRITICAL REQUIREMENTS:
1. You MUST include the complete JSON with both "name" AND "arguments"
2. The "arguments" field MUST contain all required parameters as a dictionary
3. Do NOT output only the tool name
4. Do NOT include any text before or after the <tool_call> tags
5. Do NOT include explanations, reasoning, or markdown

Your output must be ONLY the complete <tool_call> block with arguments included."""

            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": example['response']}
            ]
            return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False)}

        dataset = Dataset.from_list(data)
        return dataset.map(format_example, remove_columns=dataset.column_names)


    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, train_args_patch: Optional[Dict[str, Any]] = None):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Base defaults requested by user
        default_args = {
            "learning_rate": 2e-5,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "num_train_epochs": 5,
            "logging_steps": 5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine_with_min_lr",
            "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
            "output_dir": self.output_dir,
            "report_to": "wandb",
            "push_to_hub": False,
            "eval_steps": 100,
            "eval_strategy": "steps" if eval_dataset is not None else "no",
            "save_steps": 100,
            "save_strategy": "steps",
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "save_total_limit": 25,
        }

        # If evaluating, add the best model logic
        if eval_dataset is not None:
            default_args.update({
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
        
        if train_args_patch:
            # Handle boolean strings from CLI
            for k, v in train_args_patch.items():
                if v == "True": train_args_patch[k] = True
                if v == "False": train_args_patch[k] = False
            default_args.update(train_args_patch)

        # Pop custom keys before creating SFTConfig
        use_completion_loss = default_args.pop("completion_only_loss", True)
            
        training_args = SFTConfig(**default_args)

        # Custom callback to keep only top 10 best checkpoints
        callbacks = [TopKCheckpointerCallback(top_k=10, metric="eval_loss", greater_is_better=False)]


        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
        
        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        return os.path.join(self.output_dir, "final_model")