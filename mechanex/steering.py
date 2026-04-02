import json
import torch
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from .base import _BaseModule
from .errors import AuthenticationError, MechanexError

class SteeringVectorMethod:
    def __init__(self, model):
        self.model = model

    def check_layer_in_range(self, layer_idxs):
        # Implementation depends on model type, assuming TransformerLens/HF
        pass

class CAA(SteeringVectorMethod):
    def __call__(
        self,
        prompts: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        layer_idxs: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        if not (len(prompts) == len(positive_answers) == len(negative_answers)):
            raise ValueError("prompts, positive_answers, and negative_answers must be the same length.")

        if not layer_idxs:
            layer_idxs = []
            num_blocks = len(self.model.blocks)
            start_layer = int(num_blocks * 2 / 3)
            layer_idxs = list(range(start_layer, min(start_layer + 8, num_blocks)))
        else:
            self.check_layer_in_range(layer_idxs)

        pos_activations: Dict[int, list] = {idx: [] for idx in layer_idxs}
        neg_activations: Dict[int, list] = {idx: [] for idx in layer_idxs}

        print("Processing prompts to generate steering vectors...")
        for p, pos_answer, neg_answer in tqdm(zip(prompts, positive_answers, negative_answers), total=len(prompts)):
            # Tokenize the prompt to find its length
            prompt_tokens = self.model.to_tokens(p)
            prompt_len = prompt_tokens.shape[1]

            # Handle positive examples
            pos_example_text = p + pos_answer
            pos_tokens = self.model.to_tokens(pos_example_text)
            _, pos_cache = self.model.run_with_cache(pos_tokens, remove_batch_dim=True)
            
            for idx in layer_idxs:
                answer_activations = pos_cache["resid_post", idx][prompt_len-1:-1, :]
                if answer_activations.shape[0] == 0: continue
                # Average the activations across the answer tokens
                p_activations_mean = answer_activations.mean(dim=0).detach().cpu()
                pos_activations[idx].append(p_activations_mean)
                
            # Handle negative examples
            neg_example_text = p + neg_answer
            neg_tokens = self.model.to_tokens(neg_example_text)
            _, neg_cache = self.model.run_with_cache(neg_tokens, remove_batch_dim=True)

            for idx in layer_idxs:
                # Slice to get activations for the answer tokens
                answer_activations = neg_cache["resid_post", idx][prompt_len-1:-1, :]
                if answer_activations.shape[0] == 0: continue

                # Average the activations across the answer tokens
                n_activations_mean = answer_activations.mean(dim=0).detach().cpu()
                neg_activations[idx].append(n_activations_mean)
        
        steering_vectors = {}
        for idx in layer_idxs:
            if pos_activations[idx] and neg_activations[idx]:
                all_pos_layer = torch.stack(pos_activations[idx])
                all_neg_layer = torch.stack(neg_activations[idx])

                pos_mean = all_pos_layer.mean(dim=0)
                neg_mean = all_neg_layer.mean(dim=0)
                
                # The steering vector is the difference between the means
                vector = (pos_mean - neg_mean)
                steering_vectors[idx] = vector

        print("Steering vector computation complete.")
        return steering_vectors

class FewShot(SteeringVectorMethod):
    def __call__(
        self,
        prompts: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        layer_idxs: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        from .utils import steering_opt

        if not layer_idxs:
            layer_idxs = []
            num_blocks = len(self.model.blocks)
            start_layer = int(num_blocks * 2 / 3)
            layer_idxs = list(range(start_layer, min(start_layer + 8, num_blocks)))
        else:
            self.check_layer_in_range(layer_idxs)

        datapoints = [steering_opt.TrainingDatapoint(
            prompts[i],
            dst_completions=[positive_answers[i]],
            src_completions=[negative_answers[i]]
        ) for i in range(len(prompts))]
        steering_vectors = {}
        for layer in layer_idxs:
            vector, loss_info = steering_opt.optimize_vector(
                self.model, datapoints, layer,
                tokenizer=getattr(self.model, 'tokenizer', None),
                max_iters=20,
                lr=0.1,
                use_transformer_lens=True
            )
            print(f"Found vector for layer {layer} with loss info: {loss_info}")
            steering_vectors[layer] = vector
        return steering_vectors

class SteeringModule(_BaseModule):
    """Module for steering vector APIs."""
    def generate_vectors(self, prompts: List[str], positive_answers: List[str], negative_answers: List[str], layer_idxs: Optional[List[int]] = None, method: str = "few-shot", name: Optional[str] = None, label: Optional[str] = None) -> str:
        """
        Computes and stores steering vectors from prompts.
        Corresponds to the /steering/generate endpoint.
        Falls back to local steering if API key is missing or authentication fails.
        """
        use_local = self._client.should_use_local()

        if use_local:
            return self._generate_vectors_local(prompts, positive_answers, negative_answers, layer_idxs, method)

        try:
            if not (self._client.api_key or self._client.access_token):
                raise AuthenticationError("Authentication missing, falling back to local steering")

            resp = self._post_sse("/steering/generate", {
                "prompts": prompts,
                "positive_answers": positive_answers,
                "negative_answers": negative_answers,
                "layer_idxs": layer_idxs,
                "method": method,
                "name": name,
                "label": label
            })
            if "steering_vector_id" not in resp or resp["steering_vector_id"] is None:
                raise MechanexError(f"Steering vector ID not found in response: {resp}")
            return resp["steering_vector_id"]
        except (AuthenticationError):
            local_model = getattr(self._client, 'local_model', None)
            if local_model is not None:
                return self._generate_vectors_local(prompts, positive_answers, negative_answers, layer_idxs, method)
            raise
        except MechanexError:
            raise

    def _generate_vectors_local(
        self,
        prompts: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        layer_idxs: Optional[List[int]],
        method: str,
    ) -> str:
        local_model = getattr(self._client, 'local_model', None)
        if local_model is None:
            raise MechanexError("No local model loaded. Call mx.load_model(...) first.")

        method_normalized = (method or "few-shot").strip().lower()
        if method_normalized == "steering-perceptrons":
            raise MechanexError("Steering perceptrons are not supported for local execution.")

        steerer: Union[CAA, FewShot]
        if method_normalized == "caa":
            steerer = CAA(local_model)
        else:
            steerer = FewShot(local_model)

        vectors = steerer(prompts, positive_answers, negative_answers, layer_idxs)
        import uuid
        local_id = str(uuid.uuid4())
        if not hasattr(self._client, '_local_vectors'):
            self._client._local_vectors = {}
        self._client._local_vectors[local_id] = vectors
        return local_id
            
    def get_vectors(self, vector_id: str) -> Dict[int, torch.Tensor]:
        """
        Retrieves local steering vectors by ID.
        """
        vectors = getattr(self._client, '_local_vectors', {}).get(vector_id)
        if vectors is None:
            raise MechanexError(f"Steering vector ID '{vector_id}' not found in local session.")
        return vectors

    def save_vectors(self, vectors_or_id: Union[str, Dict[int, torch.Tensor]], path: str):
        """
        Saves steering vectors to a file.
        """
        if isinstance(vectors_or_id, str):
            # Try local first
            local_vectors = getattr(self._client, '_local_vectors', {}).get(vectors_or_id)
            if local_vectors is not None:
                vectors = local_vectors
            else:
                # Try to download from backend
                try:
                    print(f"Downloading remote steering vector '{vectors_or_id}'...")
                    resp = self._get(f"/steering/vectors/{vectors_or_id}/download")
                    # Backend returns serializable dict {layer_key: [weights]}
                    # Reconstruct it as tensors, trying to convert keys to int where possible
                    vectors = {}
                    for k, v in resp.items():
                        try:
                            key = int(k)
                        except (ValueError, TypeError):
                            key = k
                        vectors[key] = torch.tensor(v)
                except Exception as e:
                    raise MechanexError(
                        f"Steering vector '{vectors_or_id}' not found in local session "
                        f"and could not be downloaded from remote: {e}"
                    )
        else:
            vectors = vectors_or_id
        
        # Convert tensors to lists for JSON serialization
        serializable = {
            str(layer): vec.tolist() if isinstance(vec, torch.Tensor) else vec 
            for layer, vec in vectors.items()
        }
        with open(path, 'w') as f:
            json.dump(serializable, f)
        print(f"Steering vectors saved to {path}")

    def load_vectors(self, path: str) -> dict:
        """
        Loads steering vectors from a file and returns them as a dictionary.
        Keys are int when possible (local vectors), otherwise string (remote layer paths).
        """
        with open(path, 'r') as f:
            data = json.load(f)

        def _parse_key(k):
            try:
                return int(k)
            except ValueError:
                return k

        vectors = {_parse_key(layer): torch.tensor(vec) for layer, vec in data.items()}
        print(f"Steering vectors loaded from {path}")
        return vectors
            
    def generate_pairs(
        self,
        persona_name: str,
        persona_description: str = "",
        num_pairs: int = 50,
        batch_size: int = 5,
    ) -> Dict:
        """
        Generate contrastive pairs for steering vector creation.
        Uses the backend's /steering/generate-pairs endpoint.

        Args:
            persona_name: Name of the persona/behavior to generate pairs for.
            persona_description: Optional description of the persona.
            num_pairs: Number of contrastive pairs to generate.
            batch_size: Batch size for pair generation.

        Returns:
            Dict with persona, total_pairs, pairs, and avg_final_score.
        """
        return self._post("/steering/generate-pairs", {
            "persona_name": persona_name,
            "persona_description": persona_description,
            "num_pairs": num_pairs,
            "batch_size": batch_size,
        })

    def evaluate(
        self,
        steering_vector_id: str,
        positive_texts: List[str],
        negative_texts: List[str],
        test_prompts: Optional[List[str]] = None,
        persona_description: Optional[str] = None,
        strength: float = 1.0,
    ) -> Dict:
        """
        Evaluate a steering vector's effectiveness.
        Uses the backend's /steering/evaluate endpoint.

        Args:
            steering_vector_id: ID of the steering vector to evaluate.
            positive_texts: List of texts representing the desired behavior.
            negative_texts: List of texts representing the undesired behavior.
            test_prompts: Optional list of prompts to test steering on.
            persona_description: Optional description for the judge model.
            strength: Steering multiplier strength.

        Returns:
            Dict with cosine_metrics and optional judge_evaluation.
        """
        payload = {
            "steering_vector_id": steering_vector_id,
            "positive_texts": positive_texts,
            "negative_texts": negative_texts,
            "multiplier": strength,
        }
        if test_prompts is not None:
            payload["test_prompts"] = test_prompts
        if persona_description is not None:
            payload["persona_description"] = persona_description
        return self._post("/steering/evaluate", payload)

    def generate_from_jsonl(self, dataset_path: str, layer_idxs: Optional[List[int]] = None, method: str = "few-shot") -> str:
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
