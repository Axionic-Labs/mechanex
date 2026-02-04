from typing import List, Optional, Dict, Any
from .base import _BaseModule

class TrainingModule(_BaseModule):
    """
    Module for SFT and RL training pipeline APIs.
    """

    def generate_data(
        self,
        num_seeds: int = 10,
        output_file: str = "seeds.jsonl",
        topic: str = "customer relationship management",
        teacher_provider: str = "google",
        teacher_model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        teacher_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger seed data generation using a teacher model.
        
        Args:
            num_seeds: Number of prompts to generate.
            output_file: Name of the output file (e.g. "seeds.jsonl"). 
                         Stored in your tenant directory.
            topic: Broad topic for generation.
            teacher_provider: LLM provider for the teacher (e.g. "google").
            teacher_model: Specific model name for the teacher.
            api_key: Optional API key for the teacher provider.
            teacher_file: Optional path to a custom teacher implementation.
        """
        payload = {
            "num_seeds": num_seeds,
            "output_file": output_file,
            "topic": topic,
            "teacher_provider": teacher_provider,
            "teacher_model": teacher_model,
            "api_key": api_key,
            "teacher_file": teacher_file
        }
        return self._post("/mechanex_training/generate-data", payload)

    def train_sft(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        prompts_file: str = "seeds.jsonl",
        schemas_dir: str = "schemas",
        output_dir: str = "sft_output",
        teacher_model: str = "gemini-2.0-flash",
        teacher_provider: str = "google",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 8,
        use_peft: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Trigger SFT (Supervised Fine-Tuning) training.
        
        Note: All file and directory paths are relative to your tenant directory.
        
        Args:
            base_model: HuggingFace model ID or path to a base model.
            prompts_file: Path to the seed prompts file.
            schemas_dir: Directory containing tool schemas.
            output_dir: Directory where the trained model will be saved.
            teacher_model: Teacher model for trajectory generation.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Training learning rate.
            gradient_accumulation_steps: Number of steps for gradient accumulation.
            use_peft: Whether to use Parameter Efficient Fine-Tuning (LoRA).
            **kwargs: Additional hyperparameters (lora_r, lora_alpha, etc.)
        """
        payload = {
            "base_model": base_model,
            "prompts_file": prompts_file,
            "schemas_dir": schemas_dir,
            "output_dir": output_dir,
            "teacher_model": teacher_model,
            "teacher_provider": teacher_provider,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "use_peft": use_peft,
            **kwargs
        }
        return self._post("/mechanex_training/train-sft", payload)

    def train_rl(
        self,
        model_name_or_path: str = "sft_output/best_checkpoint",
        output_dir: str = "rl_output",
        prompts_file: str = "seeds.jsonl",
        schemas_dir: str = "schemas",
        teacher_model: str = "gemini-2.0-flash",
        teacher_provider: str = "google",
        num_train_epochs: int = 1,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_generations: int = 4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Trigger RL (Reinforcement Learning) training using GRPO.
        
        Note: All file and directory paths are relative to your tenant directory.
        
        Args:
            model_name_or_path: Path to the SFT-trained model.
            output_dir: Directory where the RL-trained model will be saved.
            prompts_file: Path to the prompts file for RL training.
            schemas_dir: Directory containing tool schemas.
            teacher_model: Teacher model for rewards/eval.
            num_train_epochs: Number of RL training epochs.
            batch_size: Training batch size.
            gradient_accumulation_steps: Steps for gradient accumulation.
            num_generations: Number of generations per prompt for GRPO.
            **kwargs: Additional hyperparameters.
        """
        payload = {
            "model_name_or_path": model_name_or_path,
            "output_dir": output_dir,
            "prompts_file": prompts_file,
            "schemas_dir": schemas_dir,
            "teacher_model": teacher_model,
            "teacher_provider": teacher_provider,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_generations": num_generations,
            **kwargs
        }
        return self._post("/mechanex_training/train-rl", payload)

    def run_eval(
        self,
        model_name: str,
        num_eval_samples: int = 50,
        output_dir: str = "eval_output",
        schemas_dir: str = "schemas",
        teacher_model: str = "gemini-2.0-flash",
        teacher_provider: str = "google",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Trigger model evaluation.
        
        Note: All file and directory paths are relative to your tenant directory.
        
        Args:
            model_name: Path to the model to evaluate.
            num_eval_samples: Number of fresh samples to generate for evaluation.
            output_dir: Directory where evaluation results will be saved.
            schemas_dir: Directory containing tool schemas.
            teacher_model: Teacher model for reference trajectories.
            **kwargs: Additional evaluation parameters.
        """
        payload = {
            "model_name": model_name,
            "num_eval_samples": num_eval_samples,
            "output_dir": output_dir,
            "schemas_dir": schemas_dir,
            "teacher_model": teacher_model,
            "teacher_provider": teacher_provider,
            **kwargs
        }
        return self._post("/mechanex_training/run-eval", payload)

    def deploy(
        self,
        model_path: str,
        extra_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a trained model using vLLM serve.
        
        Note: The model_path is relative to your tenant directory.
        
        Args:
            model_path: Path to the trained model to deploy.
            extra_args: Optional list of additional arguments to pass to vLLM serve.
        
        Returns:
            Response dict with task status and configuration.
        """
        payload = {
            "model_path": model_path,
            "extra_args": extra_args or []
        }
        return self._post("/mechanex_training/deploy", payload)
