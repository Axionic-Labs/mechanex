import json
from typing import List, Optional, Dict, Any, Union
from .base import _BaseModule

class SAEModule(_BaseModule):
    """
    Module for SAE-based steering and behavior management.
    """

    def create_behavior(
        self,
        behavior_name: str,
        prompts: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        description: str = "",
        steering_vector_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Define a new behavior by providing examples.
        Calculates both a steering vector (for correction) and a SAE baseline (for detection).
        
        :param behavior_name: Unique name for the behavior.
        :param prompts: List of prompt templates (e.g., ["I respond with", "My answer is"]).
        :param positive_answers: List of desired completions (e.g., [" kindness", " care"]).
        :param negative_answers: List of undesired completions (e.g., [" hate", " malice"]).
        :param description: Optional description of the behavior.
        :param steering_vector_id: Optional UUID of an existing steering vector to reuse.
        :return: JSON response containing the behavior ID, steering vector ID, and creation timestamp.
        """
        payload = {
            "behavior_name": behavior_name,
            "prompts": prompts,
            "positive_answers": positive_answers,
            "negative_answers": negative_answers,
            "description": description
        }
        if steering_vector_id:
            payload["steering_vector_id"] = steering_vector_id
            
        return self._post("/behaviors/create", payload)

    def create_behavior_from_jsonl(
        self,
        behavior_name: str,
        dataset_path: str,
        description: str = "",
        steering_vector_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Helper to create a behavior from a .jsonl file.
        Each line in the file should be a JSON object with 'prompt', 'positive_answer', and 'negative_answer' keys.

        :param behavior_name: Unique name for the behavior.
        :param dataset_path: Path to the .jsonl file.
        :param description: Optional description of the behavior.
        :param steering_vector_id: Optional UUID of an existing steering vector to reuse.
        :return: JSON response from behavior creation.
        """
        prompts, positive_answers, negative_answers = [], [], []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "prompt" in data: prompts.append(data["prompt"])
                if "positive_answer" in data: positive_answers.append(data["positive_answer"])
                if "negative_answer" in data: negative_answers.append(data["negative_answer"])
        
        return self.create_behavior(
            behavior_name=behavior_name,
            prompts=prompts,
            positive_answers=positive_answers,
            negative_answers=negative_answers,
            description=description,
            steering_vector_id=steering_vector_id
        )

    def list_behaviors(self) -> List[Dict[str, Any]]:
        """
        Returns all behaviors created by the user (scoped to API key).

        :return: List of behavior objects.
        """
        return self._get("/behaviors")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        behavior_names: Optional[List[str]] = None,
        auto_correct: bool = True,
        force_steering: Optional[List[str]] = None
    ) -> str:
        """
        Utilizes Sparse Autoencoders (SAE) to detect and correct behavioral drift 
        or target specific behaviors during generation.

        :param prompt: The input prompt.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param behavior_names: Optional list of specific behavior names to monitor.
        :param auto_correct: Whether to automatically correct detected behaviors.
        :param force_steering: Optional list of behavior names to force steering on.
        :return: The generated text string.
        """
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "behavior_names": behavior_names,
            "auto_correct": auto_correct,
            "force_steering": force_steering
        }
        response = self._post("/sae/generate", payload)
        return response.get("text", "")
