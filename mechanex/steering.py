import json
from typing import List, Optional
from .base import _BaseModule

class SteeringModule(_BaseModule):
    """Module for steering vector APIs."""
    def generate_vectors(self, prompts: List[str], positive_answers: List[str], negative_answers: List[str], layer_idxs: Optional[List[int]] = None, method: str = "few-shot") -> dict:
        """
        Computes and stores steering vectors from prompts.
        Corresponds to the /steering/generate endpoint.
        """
        return self._post("/steering/generate", {
            "prompts": prompts,
            "positive_answers": positive_answers,
            "negative_answers": negative_answers,
            "layer_idxs": layer_idxs,
            "method": method
        })
        
    def generate_from_jsonl(self, dataset_path: str, layer_idxs: Optional[List[int]] = None, method: str = "few-shot") -> dict:
        """
        A helper to generate steering vectors from a .jsonl file.
        Each line in the file should be a JSON object with 'positive' and 'negative' keys.
        """
        positive, negative, prompts = [], [], []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "prompt" in data: prompts.append(data["prompt"])
                if "positive_answer" in data: positive.append(data["positive_answer"])
                if "negative_answer" in data: negative.append(data["negative_answer"])
        return self.generate_vectors(prompts, positive, negative, layer_idxs, method)