<a id="mechanex"></a>

# mechanex

Mechanex: A Python client for the Axionic API.

<a id="mechanex.generation"></a>

# mechanex.generation

<a id="mechanex.generation.GenerationModule"></a>

## GenerationModule Objects

```python
class GenerationModule(_BaseModule)
```

<a id="mechanex.generation.GenerationModule.generate"></a>

#### generate

```python
def generate(prompt: str,
             max_tokens: int = 128,
             sampling_method: str = "top-k",
             top_k: int = 50,
             top_p: float = 0.9,
             steering_strength: float = 0,
             steering_vector=None) -> str
```

Runs a standard generation. Falls back to local model if available.

<a id="mechanex.serving"></a>

# mechanex.serving

<a id="mechanex.sae"></a>

# mechanex.sae

<a id="mechanex.sae.SAEModule"></a>

## SAEModule Objects

```python
class SAEModule(_BaseModule)
```

Module for SAE-based steering and behavior management.

<a id="mechanex.sae.SAEModule.create_behavior"></a>

#### create\_behavior

```python
def create_behavior(
        behavior_name: str,
        prompts: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        description: str = "",
        steering_vector_id: Optional[str] = None) -> Dict[str, Any]
```

Define a new behavior. Falls back to local computation if remote fails.

<a id="mechanex.sae.SAEModule.create_behavior_from_jsonl"></a>

#### create\_behavior\_from\_jsonl

```python
def create_behavior_from_jsonl(
        behavior_name: str,
        dataset_path: str,
        description: str = "",
        steering_vector_id: Optional[str] = None) -> Dict[str, Any]
```

Helper to create a behavior from a .jsonl file.

<a id="mechanex.sae.SAEModule.list_behaviors"></a>

#### list\_behaviors

```python
def list_behaviors() -> List[Dict[str, Any]]
```

Returns behaviors. Combines remote and local if available.

<a id="mechanex.sae.SAEModule.generate"></a>

#### generate

```python
def generate(prompt: str,
             max_new_tokens: int = 50,
             behavior_names: Optional[List[str]] = None,
             force_steering: Optional[List[str]] = None) -> str
```

Generation with SAE monitoring. Falls back to local if needed.

<a id="mechanex.client"></a>

# mechanex.client

<a id="mechanex.client.Mechanex"></a>

## Mechanex Objects

```python
class Mechanex()
```

A client for interacting with the Axionic API.

<a id="mechanex.client.Mechanex.signup"></a>

#### signup

```python
def signup(email, password)
```

Register a new user.

<a id="mechanex.client.Mechanex.login"></a>

#### login

```python
def login(email, password)
```

Authenticate and set API key.

<a id="mechanex.client.Mechanex.list_api_keys"></a>

#### list\_api\_keys

```python
def list_api_keys()
```

List API keys for the current user.

<a id="mechanex.client.Mechanex.create_api_key"></a>

#### create\_api\_key

```python
def create_api_key(name: str = "Default Key")
```

Create a new API key for the current user.

<a id="mechanex.client.Mechanex.whoami"></a>

#### whoami

```python
def whoami()
```

Get current user information.

<a id="mechanex.client.Mechanex.serve"></a>

#### serve

```python
def serve(model=None,
          host="0.0.0.0",
          port=8000,
          use_vllm=False,
          corrected_behaviors: Optional[List[str]] = None)
```

Turn the model into an OpenAI compatible endpoint.

<a id="mechanex.client.Mechanex.set_local_model"></a>

#### set\_local\_model

```python
def set_local_model(model)
```

Set a local model (e.g. TransformerLens) for local steering fallback.

<a id="mechanex.client.Mechanex.load"></a>

#### load

```python
def load(model_name: str, **kwargs) -> 'Mechanex'
```

Alias for load_model.

<a id="mechanex.client.Mechanex.unload"></a>

#### unload

```python
def unload() -> 'Mechanex'
```

Alias for unload_model.

<a id="mechanex.client.Mechanex.unload_model"></a>

#### unload\_model

```python
def unload_model() -> 'Mechanex'
```

Unloads the local model and clears associated metadata.

<a id="mechanex.client.Mechanex.load_model"></a>

#### load\_model

```python
def load_model(model_name: str, **kwargs) -> 'Mechanex'
```

Loads a model locally using TransformerLens and automatically configures SAE settings.

<a id="mechanex.client.Mechanex.get_huggingface_models"></a>

#### get\_huggingface\_models

```python
@staticmethod
def get_huggingface_models(host: str = "127.0.0.1",
                           port: int = 8000) -> List[str]
```

Fetches the list of available public models from Hugging Face.
This is a static method and does not require a model to be loaded.

<a id="mechanex.steering"></a>

# mechanex.steering

<a id="mechanex.steering.SteeringModule"></a>

## SteeringModule Objects

```python
class SteeringModule(_BaseModule)
```

Module for steering vector APIs.

<a id="mechanex.steering.SteeringModule.generate_vectors"></a>

#### generate\_vectors

```python
def generate_vectors(prompts: List[str],
                     positive_answers: List[str],
                     negative_answers: List[str],
                     layer_idxs: Optional[List[int]] = None,
                     method: str = "few-shot") -> str
```

Computes and stores steering vectors from prompts.
Corresponds to the /steering/generate endpoint.
Falls back to local steering if API key is missing or authentication fails.

<a id="mechanex.steering.SteeringModule.get_vectors"></a>

#### get\_vectors

```python
def get_vectors(vector_id: str) -> Dict[int, torch.Tensor]
```

Retrieves local steering vectors by ID.

<a id="mechanex.steering.SteeringModule.save_vectors"></a>

#### save\_vectors

```python
def save_vectors(vectors_or_id: Union[str, Dict[int, torch.Tensor]],
                 path: str)
```

Saves steering vectors to a file.

<a id="mechanex.steering.SteeringModule.load_vectors"></a>

#### load\_vectors

```python
def load_vectors(path: str) -> Dict[int, torch.Tensor]
```

Loads steering vectors from a file and returns them as a dictionary.

<a id="mechanex.steering.SteeringModule.generate_from_jsonl"></a>

#### generate\_from\_jsonl

```python
def generate_from_jsonl(dataset_path: str,
                        layer_idxs: Optional[List[int]] = None,
                        method: str = "few-shot") -> str
```

A helper to generate steering vectors from a .jsonl file.
Each line in the file should be a JSON object with 'positive' and 'negative' keys.

<a id="mechanex.utils"></a>

# mechanex.utils

<a id="mechanex.utils.steering_opt"></a>

# mechanex.utils.steering\_opt

<a id="mechanex.utils.steering_opt.hf_hooks_contextmanager"></a>

#### hf\_hooks\_contextmanager

```python
@contextmanager
def hf_hooks_contextmanager(model, hook_infos: List[Tuple[int, Callable]])
```

A context manager for running a HuggingFace Llama-like model with hooks (particularly steering hooks).

**Arguments**:

- `model` _HuggingFace model_ - the model to hook into
- `hook_infos` - a list of pairs. The first element of each pair is the layer to hook into, and the second element is the hook function to attach.
  

**Example**:

  # make and apply a steering hook to a HuggingFace model
  layer = 10
  hook_fn = steering_opt.make_steering_hook_hf(vector)
  # generate tokens while hooked
  with steering_opt.hf_hooks_contextmanager(model, [(layer, hook_fn)]):
  input_tokens = tokenizer("Hello, world.", return_tensors="pt")
  generated_tokens = model.generate(**input_tokens, max_new_tokens=10)
  # print generated tokens
  print(tokenizer.batch_decode(generated_tokens)[0])

<a id="mechanex.utils.steering_opt.make_abl_mat"></a>

#### make\_abl\_mat

```python
def make_abl_mat(v)
```

Makes a matrix M from a vector v such that applying M to a vector x ablates the component of x in the direction of v (i.e. projects x onto the orthogonal complement of v).

This is useful for ablation steering and clamp steering (see Sec. 2 of https://arxiv.org/pdf/2411.09003), where we want to remove all information in the direction of v from model activations x (ablation steering), or where we want to set the component of x in the direction of v to have a certain value (clamp steering).

For example, the following steering hook clamps the component of x in the direction of v to be 5:
`steering_opt.make_steering_hook_hf(5*v, make_abl_mat(v))`

<a id="mechanex.utils.steering_opt.make_steering_hook_hf"></a>

#### make\_steering\_hook\_hf

```python
def make_steering_hook_hf(vector_, matrix=None, token=None)
```

Makes a hook for steering the activations of a HuggingFace model.

**Arguments**:

- `vector_` - a vector which will be added to the activations
- `matrix` _optional_ - a matrix, such that the product of that matrix with the activations will be added to the activations
- `token` _optional_ - an int or a slice denoting which tokens to apply steering to.

<a id="mechanex.utils.steering_opt.make_steering_hook_tflens"></a>

#### make\_steering\_hook\_tflens

```python
def make_steering_hook_tflens(vector, matrix=None, token=None)
```

Makes a hook for steering the activations of a TransformerLens model.

**Arguments**:

- `vector_` - a vector which will be added to the activations
- `matrix` _optional_ - a matrix, such that the product of that matrix with the activations will be added to the activations
- `token` _optional_ - an int or a slice denoting which tokens to apply steering to.

<a id="mechanex.utils.steering_opt.make_activs_hook_hf"></a>

#### make\_activs\_hook\_hf

```python
def make_activs_hook_hf(outlist)
```

Makes a hook for storing the activations of a HuggingFace model.

**Arguments**:

- `outlist` _list_ - a list to which the activations of the model will be appended

<a id="mechanex.utils.steering_opt.get_completion_logprob"></a>

#### get\_completion\_logprob

```python
def get_completion_logprob(model,
                           prompt,
                           completion,
                           tokenizer=None,
                           coldness=1,
                           return_all_probs=False,
                           do_one_minus=False,
                           do_log=True,
                           eps=0,
                           use_transformer_lens=True,
                           **kwargs)
```

Gets the model's log probabilities of a completion for a prompt.

**Arguments**:

- `model` - the model to be used
- `prompt` _str_ - the input prompt to the model
- `completion` _str_ - the completion whose probability is to be obtained
- `tokenizer` _required for HuggingFace models_ - The tokenizer associated with the model
- `coldness` _float, 1 by default_ - The coldness/inverse temperature parameter used in computing probabilities
- `return_all_probs` _False by default_ - If True, then return the probabilities for each token. Otherwise, only return the joint probability of the whole sequence.
- `do_one_minus` _False by default_ - If True, then take the probability of the complement of the completion.
- `do_log` _True by default_ - If True, then use log probabilities (base 10).
- `eps` _float, 0 by default_ - Used to avoid underflow errors.
- `use_transformer_lens` _True by default_ - If True, then the model is a TransformerLens model. Otherwise, the model is a HuggingFace model. Note: for HuggingFace models, one can use the wrapper get_completion_logprob_hf().
- `**kwargs` - additional keyword arguments passed to the model.
  

**Returns**:

  If return_all_probs is False, then returns the joint (log) probability of the sequence. Otherwise, returns a tuple containing the joint (log) probability of the sequence and the (log) probability of each token

<a id="mechanex.utils.steering_opt.get_completion_logprob_hf"></a>

#### get\_completion\_logprob\_hf

```python
def get_completion_logprob_hf(model, prompt, completion, tokenizer, **kwargs)
```

Gets a HuggingFace model's log probabilities of a completion for a prompt.

**Arguments**:

- `model` - the model to be used
- `prompt` _str_ - the input prompt to the model
- `completion` _str_ - the completion whose probability is to be obtained
- `tokenizer` - the tokenizer associated with the model
- `coldness` _float, 1 by default_ - The coldness/inverse temperature parameter used in computing probabilities
- `return_all_probs` _False by default_ - If True, then return the probabilities for each token. Otherwise, only return the joint probability of the whole sequence.
- `do_one_minus` _False by default_ - If True, then take the probability of the complement of the completion.
- `do_log` _True by default_ - If True, then use log probabilities.
- `eps` _float, 0 by default_ - Used to avoid underflow errors.
- `**kwargs` - additional keyword arguments passed to the model.
  

**Returns**:

  If return_all_probs is False, then returns the joint (log) probability of the sequence. Otherwise, returns a tuple containing the joint (log) probability of the sequence and the (log) probability of each token

<a id="mechanex.utils.steering_opt.sample_most_likely_completions_hf"></a>

#### sample\_most\_likely\_completions\_hf

```python
@torch.no_grad()
def sample_most_likely_completions_hf(model,
                                      tokenizer,
                                      dst_prompt,
                                      src_prompt=None,
                                      k=5,
                                      iters=5,
                                      coldness=1,
                                      do_one_minus=False,
                                      gc_interval=3,
                                      use_total_probs=False,
                                      reverse=False,
                                      return_log_probs=False,
                                      return_token_probs=True,
                                      **kwargs)
```

Performs greedy beam search sampling for a HuggingFace model.

**Arguments**:

- `model` - the model to be used
- `tokenizer` - the tokenizer for the model
- `dst_prompt` _str_ - the prompt given as input to the model, whose completions are to be sampled
- `src_prompt` _optional, str_ - if this prompt is given, then the returned completions will be those that maximize the *difference* in probability between when dst_prompt is used as the input versus when src_prompt is used.
- `k` _int, 5 by default_ - the number of beams (and the number of completions to return).
- `iters` _int, 5 by default_ - the number of tokens to sample.
- `coldness` _float, 1 by default_ - the coldness/inverse temperature parameter that affects the entropy of the model's distribution.
- `do_one_minus` _False by default_ - if True, then when computing completion probabilities, take the probability of the complement of the completion. (This means that the function will return the least-likely completions. This is useful in conjunction with src_prompt.)
- `gc_interval` _int, 3 by default_ - every gc_interval iterations, the garbage collector will be run to prevent OOMs.
- `use_total_probs` _False by default_ - if True, then return the joint probability of each completion rather than the probability of each individual token in each completion.
- `reverse` _False by default_ - if True, then return the least likely completions (similar to do_one_minus).
- `return_log_probs` _False by default_ - if True, then return log (base 10) probabilities of completions.
- `**kwargs` - additional keyword arguments that will be passed to the model.
  

**Returns**:

- `completions` _list of strs_ - a list of the k sampled completions.
- `completion_probs` _list_ - if use_total_probs is True, then a list of the joint (log) probabilities of each completion. Otherwise, a list of lists, one for each completion, where each inner list contains the (log) probability of each token in that completion.

<a id="mechanex.utils.steering_opt.TrainingDatapoint"></a>

## TrainingDatapoint Objects

```python
@dataclasses.dataclass
class TrainingDatapoint()
```

A datapoint used for optimizing steering vectors.

Members:
	prompt (str): the prompt used in this datapoint
	src_completions (optional, list of strs): a list of completions whose probabilities on this prompt should be minimized by the steering vector (i.e. suppression steering targets)
	dst_completions (optional, list of strs): a list of completions whose probabilities on this prompt should be maximized by the steering vector (i.e. promotion steering targets)
	src_completions_target_losses (optional, list of floats): a list of target losses for each suppression steering target, such that if all targets' losses fall below their respective given target losses, the optimization process will stop early
	dst_completions_target_losses (optional, list of floats): a list of target losses for each promotion steering target, such that if all targets' losses fall below their respective given target losses, the optimization process will stop early
	token (optional, slice or int): the tokens in this prompt that steering should be applied to when optimizing on this datapoint. If not given, then steering will be applied to all tokens.
	is_negative (False by default): if True, then the vector being optimized will be negated on this datapoint

<a id="mechanex.utils.steering_opt.optimize_vector"></a>

#### optimize\_vector

```python
def optimize_vector(model,
                    datapoints,
                    layer,
                    eps=1e-6,
                    lr=0.01,
                    max_iters=None,
                    coldness=0.7,
                    normalize_token_length=False,
                    only_hook_prompt=False,
                    use_transformer_lens=False,
                    tokenizer=None,
                    target_loss=None,
                    return_info=True,
                    do_target_loss_sum=True,
                    return_loss_history=False,
                    return_vec_history=False,
                    target_loss_target_iters=1,
                    satisfice=False,
                    do_one_minus=True,
                    max_norm=None,
                    starting_norm=1,
                    starting_vec=None,
                    vector_clamp=None,
                    affine_rank=None,
                    max_affine_norm=2,
                    starting_affine_norm=1,
                    noise_scale=None,
                    do_tangent_space_noise=True,
                    do_noise_abl_relu=False,
                    noise_iters=1,
                    do_antipgd=False,
                    do_output_constr=False,
                    custom_output_constr_loss_func=None,
                    custom_output_constr_pre_loss_func=None,
                    output_constr_norm_initial_scale=1,
                    output_constr_lr=None,
                    max_output_constr_iters=None,
                    debug=False)
```

Optimize a steering vector on a set of datapoints.

**Arguments**:

  Required args:
- `model` - the model to optimize the steering vector for
- `datapoints` _list of TrainingDatapoints_ - the list of TrainingDatapoints to optimize over
- `layer` _int or list of ints_ - the layer(s) to apply the steering vector to. If an int, then only optimize the steering vector at that layer. Otherwise, optimize the steering vector at all layers in the list.
  
  HuggingFace-related args:
- `use_transformer_lens` _False by default_ - set to True if the model being used is a TransformerLens model. If the model is a HuggingFace model, then set to False (and pass a value for tokenizer).
- `tokenizer` _required for HuggingFace models_ - the tokenizer associated with the HuggingFace model being used
  
  General hyperparams:
- `eps` _float, 1e-6 by default_ - a small constant used to prevent underflow errors
- `lr` _float, 0.01 by default_ - the learning rate for the optimizer
- `coldness` _float, 0.7 by default_ - the coldness/inverse temperature parameter used for computing probabilities
  
  Early stopping and loss-related args:
- `max_iters` _int, optional_ - if set, then terminate optimization after this many steps.
- `target_loss` _float, optional_ - if set, then stop the optimization when the loss stays below target_loss for target_loss_target_iters steps.
- `do_target_loss_sum` _True by default_ - used with target_loss. If True, then stop optimization when the sum of losses on all completions is below target_loss. If False, then stop optimization when each completion's loss is individually below target_loss.
- `target_loss_target_iters` _int, 1 by default_ - used for early stopping. If the loss stays below target_loss for this many optimization steps, or the absolute difference in loss from the previous step to the current step stays below eps for this many steps, then exit the optimization loop early.
- `satisfice` _False by default_ - if True, then penalize the vector based on the squared difference between the actual loss and target_loss.
- `normalize_completion_length` _False by default_ - if True, then divide the loss for each completion by the number of tokens in the completion.
- `do_one_minus` _True by default_ - if True, then for src completions, compute loss using the log of one minus the probability of each completion. If False, then for src completions, compute loss based on the negative log probability of each completion.
  
  Return value options:
- `return_info` _True by default_ - if True, then in addition to the steering vector itself, return a dictionary containing info about the optimization process.
- `return_loss_history` _False by default_ - if True, then return a list of losses after each optimization step.
- `return_vec_history` _False by default_ - if True, then return a list containing the steering vector after each optimization step.
  
  Misc args:
- `only_hook_prompt` _False by default_ - if True, then only apply the steering vector to tokens in the prompt (rather than all tokens, including those in the completion).
- `debug` _False by default_ - if True, then print out loss information at each optimization step.
  
  Norm-constrained steering args:
- `max_norm` _float, optional_ - the maximum norm of the steering vector. If set, then after each optimization step, if the vector's norm exceeds max_norm, it will be rescaled to max_norm.
- `starting_norm` _float, 1 by default_ - the starting norm of the steering vector. Before optimization, the steering vector is initialized to a randomly spherically-distributed vector with this norm.
- `starting_vec` _optional_ - if given, then this vector is used to initialize the vector being optimized, instead of the default random optimization.
  
  Clamp/affine steering args:
- `vector_clamp` _float, optional_ - if set, then optimize the vector to perform clamp steering. For a vector v, clamp steering with v on activations x entails ablating the component of x in the v direction, and then adding vector_clamp*v to the resulting activations.
- `affine_rank` _int, optional_ - if set, then perform affine steering, which optimizes a low-rank *steering matrix* in addition to a steering vector. For vector v and matrix M, affine steering with v and M on activations x entails mapping x to x + Mx + v. affine_rank is the rank of the matrix M.
- `max_affine_norm` _int, 2 by default_ - the low-rank steering matrix is internally factorized into two matrices M_l and M_r such that M = M_l^T M_r. After each optimization step, if the norm of any of the columns of M_l and M_r is greater than max_affine_norm, then that column is rescaled to have norm max_affine_norm. (This approach, instead of using e.g. spectral norm, was inspired by MELBO.)
- `starting_affine_norm` _int, 1 by default_ - the norm of the columns of M_l and M_r upon initialization.
  
  Noisy steering args:
- `noise_scale` _float, optional_ - if set, then add Gaussian noise multiplied by noise_scale to the activations at each optimization step (as a form of regularization).
- `do_tangent_space_noise` _True by default_ - if True, then project the noise vector onto the tangent space of the loss w.r.t. the steering vector. (Ideally, an approximation to prevent noise from inducing instability in the loss.)
- `do_noise_abl_relu` _False by default_ - only takes effect when do_tangent_space_noise is set. If True, then only ablates the component of the noise vector that points in the direction of decreasing loss (i.e. don't do ablation if the noise vector points in the direction of increasing loss). Ideally, this approximates choosing noise that only ever increases the loss or keeps it the same.
- `noise_iters` _1 by default_ - how many times noise should be sampled at each optimization step before updating the steering vector.
- `do_antipgd` _False by default_ - if True, then uses anti-correlated noise (see https://proceedings.mlr.press/v162/orvieto22a/orvieto22a.pdf)
  
  Output-constrained steering args:
- `do_output_constr` _False by default_ - if True, then perform output-constrained steering. This entails the following: after finding a steering vector that satisfies the target loss, then perform constrained minimization to optimize a vector with the smallest norm that does not increase the loss.
- `output_constr_lr` _float, optional_ - if set, then use this learning rate when performing output-constrained optimization instead of the learning rate used for the base steering optimization phase.
- `max_output_constr_iters` _int, optional_ - if set, then terminate output-constrained optimization after this many steps
  
- `custom_output_constr_loss_func` _function, optional, only supports TransformerLens_ - if set, then during the output-constrained phase, instead of minimizing the vector norm, minimize this loss function (while also ensuring that the vector's norm doesn't increase beyond its initial value). The function should have the following signature: custom_output_constr_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt, **kwargs). In this function, model, datapoints, layer, vector, and only_hook_prompt are the same as those passed to optimize_vector(). matrix_left and matrix_right are the two factor matrices used in affine steering if affine_rank is not None; otherwise, they are None. custom_output_constr_loss_func() can take optional kwargs, which will be set to the result of running custom_output_constr_pre_loss_func() before the first output-constrained optimization step (if custom_output_constr_pre_loss_func is not None). This function should return a scalar PyTorch tensor which can be backpropagated.
- `custom_output_constr_pre_loss_func` _function, optional, only supports TransformerLens_ - if set, then this function will be run before the first output-constrained optimization step. It should return a dictionary, which will be passed as kwargs to custom_output_constr_loss_func(). This function should have the signature custom_output_constr_pre_loss_func(model, datapoints, layer, vector, matrix_left, matrix_right, only_hook_prompt).
- `output_constr_norm_initial_scale` _float, 1 by default, only supports TransformerLens_ - only used with custom output-constrained loss functions. This is the amount by which the norm constraint (which prevents the vector's norm from increasing beyond its initial value) will be scaled. Constrained optimization works best when the constraints have similar scale, so this parameter allows the norm constraint to be placed at a similar scale to the custom loss function.
  
  Returns a tuple containing the following elements in order:
- `vector` - the steering vector which has been optimized
- `matrix` _optional_ - if affine_norm is not None, then the affine steering matrix which has been optimized
- `info` _optional_ - if return_info is True, then a dictionary containing the following items:
- `iters` - the number of optimization steps taken
- `norm` - the norm of the returned vector
- `loss` - if do_target_loss_sum is True, then the sum of losses. Otherwise, individual completion losses, and if return_loss_history is True, individual completion losses for all optimization steps are returned.
- `vec_history` _optional_ - if return_vec_history is True, then a list containing the steering vector after each optimization step
- `output_constr_iters` _optional_ - if output-constrained optimization was performed, then the number of optimization steps taken during the output-constrained optimization phase

<a id="mechanex.utils.steering_opt.make_melbo_loss_funcs"></a>

#### make\_melbo\_loss\_funcs

```python
def make_melbo_loss_funcs(target_layer)
```

Make custom loss functions for performing MELBO (www.lesswrong.com/posts/ioPnHKFyy4Cw2Gr2x) in conjunction with output-constrained optimization. Only supports TransformerLens.

**Arguments**:

- `target_layer` - the layer at which the distance between steered and unsteered activations will be used to compute the MELBO loss.
  

**Returns**:

  melbo_pre_loss_func, melbo_loss_func: two functions which should be passed to optimize_vector() as custom_output_constr_pre_loss_func and custom_output_constr_loss_func respectively

<a id="mechanex.utils.steering_opt.optimize_vector_minibatch_hf"></a>

#### optimize\_vector\_minibatch\_hf

```python
def optimize_vector_minibatch_hf(model,
                                 tokenizer,
                                 prompts,
                                 layer,
                                 src_completions=None,
                                 dst_completions=None,
                                 minibatch_size=5,
                                 eps=1e-6,
                                 lr=0.01,
                                 max_iters=None,
                                 coldness=0.7,
                                 target_loss=None,
                                 target_loss_target_iters=1,
                                 satisfice=False,
                                 starting_norm=1,
                                 max_norm=None,
                                 affine_rank=None,
                                 max_affine_norm=None,
                                 debug=False,
                                 return_info=True,
                                 vector_clamp=None)
```

An alternative version to optimize_vector() that uses minibatching to speed up optimization. More limited than optimize_vector(), but faster. Only supports HuggingFace.

**Arguments**:

- `model` - the HuggingFace model to optimize for
- `tokenizer` - the associated tokenizer
- `prompts` - a list of prompts to optimize over
- `src_completions` - a list of completions to suppress. All src_completions will be optimized over for all prompts.
- `dst_completions` - a list of completions to promote. All dst_completions will be optimized over for all prompts.
  
  All other arguments are the same as in optimize_vector() (although note that there are many arguments to optimize_vector() that are not supported by optimize_vector_minibatch_hf()).
  
  Returns a tuple containing the following elements in order:
- `vector` - the steering vector which has been optimized
- `matrix` - if affine_norm is not None, then the affine steering matrix which has been optimized
- `info` - if return_info is True, then a dictionary containing the following items:
- `iters` - the number of optimization steps taken
- `norm` - the norm of the returned vector
- `loss` - the total loss at the end of optimization

<a id="mechanex.model"></a>

# mechanex.model

<a id="mechanex.model.ModelModule"></a>

## ModelModule Objects

```python
class ModelModule(_BaseModule)
```

Module for inspecting the model structure.

<a id="mechanex.model.ModelModule.get_graph"></a>

#### get\_graph

```python
def get_graph() -> List[Dict[str, Any]]
```

Retrieves the model's computation graph.
Corresponds to the /graph endpoint.

<a id="mechanex.model.ModelModule.get_paths"></a>

#### get\_paths

```python
def get_paths() -> List[str]
```

Retrieves all available layer paths in the model.
Corresponds to the /paths endpoint.

<a id="mechanex.raag"></a>

# mechanex.raag

<a id="mechanex.raag.truncate_text"></a>

#### truncate\_text

```python
def truncate_text(text, max_width, font_properties)
```

Truncates text with '...' if it exceeds the max_width.

<a id="mechanex.raag.add_glow_effect"></a>

#### add\_glow\_effect

```python
def add_glow_effect(patch,
                    ax,
                    glow_color,
                    n_layers=10,
                    max_alpha=0.3,
                    diff_linewidth=1.5)
```

Adds a glow effect to a matplotlib patch.

<a id="mechanex.raag.RAAGModule"></a>

## RAAGModule Objects

```python
class RAAGModule(_BaseModule)
```

Module for Retrieval-Augmented Answer Generation APIs.

<a id="mechanex.raag.RAAGModule.generate"></a>

#### generate

```python
def generate(qa_entries: List[dict],
             docs: List[dict] = None,
             pinecone_index_name: str = None) -> dict
```

Performs Retrieval-Augmented Answer Generation.
Corresponds to the /raag/generate endpoint.

<a id="mechanex.attribution"></a>

# mechanex.attribution

<a id="mechanex.attribution.AttributionModule"></a>

## AttributionModule Objects

```python
class AttributionModule(_BaseModule)
```

Module for attribution patching APIs.

<a id="mechanex.attribution.AttributionModule.compute_scores"></a>

#### compute\_scores

```python
def compute_scores(clean_prompt: str,
                   corrupted_prompt: str,
                   target_module_paths: Optional[List[str]] = None) -> dict
```

Computes attribution scores by patching the model.
Corresponds to the /attribution-patching/scores endpoint.

<a id="mechanex.cli"></a>

# mechanex.cli

<a id="mechanex.cli.main"></a>

#### main

```python
@click.group()
@click.pass_context
def main(ctx)
```

Mechanex CLI for managing your Axionic account and models.

<a id="mechanex.cli.signup"></a>

#### signup

```python
@main.command()
@click.option('--email', prompt='Email', help='Your email address.')
@click.option('--password',
              prompt=True,
              hide_input=True,
              help='Your password.')
@click.pass_obj
def signup(obj, email, password)
```

Sign up for a new Axionic account, log in, and generate an API key.

<a id="mechanex.cli.login"></a>

#### login

```python
@main.command()
@click.option('--email', prompt='Email', help='Your email address.')
@click.option('--password',
              prompt=True,
              hide_input=True,
              help='Your password.')
@click.pass_obj
def login(obj, email, password)
```

Log in to your Axionic account.

<a id="mechanex.cli.list_api_keys"></a>

#### list\_api\_keys

```python
@main.command()
def list_api_keys()
```

List your Axionic API keys.

<a id="mechanex.cli.create_api_key"></a>

#### create\_api\_key

```python
@main.command()
@click.option('--name',
              default='Default Key',
              help='Name for the new API key.')
def create_api_key(name)
```

Create a new Axionic API key.

<a id="mechanex.cli.whoami"></a>

#### whoami

```python
@main.command()
def whoami()
```

Show the current logged-in user and profile info.

<a id="mechanex.cli.logout"></a>

#### logout

```python
@main.command()
def logout()
```

Log out and remove stored credentials.

<a id="mechanex.errors"></a>

# mechanex.errors

<a id="mechanex.errors.MechanexError"></a>

## MechanexError Objects

```python
class MechanexError(Exception)
```

Base exception for the Mechanex library.

<a id="mechanex.errors.AuthenticationError"></a>

## AuthenticationError Objects

```python
class AuthenticationError(MechanexError)
```

Raised when authentication fails.

<a id="mechanex.errors.APIError"></a>

## APIError Objects

```python
class APIError(MechanexError)
```

Raised when the API returns an error.

<a id="mechanex.errors.NotFoundError"></a>

## NotFoundError Objects

```python
class NotFoundError(MechanexError)
```

Raised when a resource is not found.

<a id="mechanex.errors.RateLimitError"></a>

## RateLimitError Objects

```python
class RateLimitError(MechanexError)
```

Raised when rate limits are exceeded.

<a id="mechanex.errors.ValidationError"></a>

## ValidationError Objects

```python
class ValidationError(MechanexError)
```

Raised when request validation fails.

<a id="mechanex.base"></a>

# mechanex.base

<a id="mechanex.base._BaseModule"></a>

## \_BaseModule Objects

```python
class _BaseModule()
```

A base class for API modules to handle requests, errors, and authentication.

<a id="mechanex.base._BaseModule.__init__"></a>

#### \_\_init\_\_

```python
def __init__(client)
```

Initialize the module.

**Arguments**:

- `client`: The main client instance.
- `api_key`: Optional API key for Authorization.

