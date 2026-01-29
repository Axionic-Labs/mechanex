import json
import torch
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from .base import _BaseModule
from .errors import AuthenticationError, MechanexError
from .utils import steering_opt

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
        layer_idxs: List[int] = None
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

        pos_activations = {idx: [] for idx in layer_idxs}
        neg_activations = {idx: [] for idx in layer_idxs}

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
        layer_idxs: List[int] = None
    ) -> Dict[int, torch.Tensor]:

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
    def generate_vectors(self, prompts: List[str], positive_answers: List[str], negative_answers: List[str], layer_idxs: Optional[List[int]] = None, method: str = "few-shot") -> str:
        """
        Computes and stores steering vectors from prompts.
        Corresponds to the /steering/generate endpoint.
        Falls back to local steering if API key is missing or authentication fails.
        """
        if not self._client.api_key:
            raise MechanexError("API key required. Call mx.set_key() first.")

        try:
            if self._client.api_key is None:
                raise AuthenticationError("API key missing, falling back to local steering")

            resp = self._post("/steering/generate", {
                "prompts": prompts,
                "positive_answers": positive_answers,
                "negative_answers": negative_answers,
                "layer_idxs": layer_idxs,
                "method": method
            })
            return resp["steering_vector_id"]
        except (AuthenticationError, MechanexError) as e:
            # Check if we have a local model to use for fallback
            local_model = getattr(self._client, 'local_model', None)
            if local_model is not None:
                if method == "caa":
                    steerer = CAA(local_model)
                else:
                    steerer = FewShot(local_model)
                
                vectors = steerer(prompts, positive_answers, negative_answers, layer_idxs)
                # Store vectors locally or return a local reference
                # For now we'll just return a placeholder ID and store them on the client
                import uuid
                local_id = str(uuid.uuid4())
                if not hasattr(self._client, '_local_vectors'):
                    self._client._local_vectors = {}
                self._client._local_vectors[local_id] = vectors
                return local_id
            else:
                raise e
            
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
            vectors = self.get_vectors(vectors_or_id)
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

    def load_vectors(self, path: str) -> Dict[int, torch.Tensor]:
        """
        Loads steering vectors from a file and returns them as a dictionary.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        vectors = {int(layer): torch.tensor(vec) for layer, vec in data.items()}
        print(f"Steering vectors loaded from {path}")
        return vectors
            
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