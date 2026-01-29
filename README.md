# Mechanex

Mechanex allows you to control and debug your LLMs. Learn more at [axioniclabs.ai](https://axioniclabs.ai/)

## Installation

```bash
pip install mechanex
```

## Quick Start

### 1. Initialize the Client
You must authenticate before using any features. You can do this by running `mechanex login` in your terminal (which saves your **session token** globally), or by setting an API key manually in your code.

**Python API:**
```python
import mechanex as mx
mx.set_key("your-api-key-here")
```

**CLI:**
Authenticate via user login to enable usage in all your scripts without manual configuration:
```bash
mechanex login
```

### 2. Standard Generation

```python
output = mx.generation.generate(
    prompt="The future of AI is",
    max_tokens=50,
    sampling_method="top-k" # Options: greedy, top-k, top-p, ads
)
print(output)
```

## Local Model Management

Mechanex allows you to load models locally for inspection and low-latency hooks.

### Loading a Local Model
```python
import mechanex as mx
mx.set_key("your-api-key-here") # Required even for local mode

mx.load("gpt2") # Uses transformer-lens to load the model
```

### Unloading a Model
To free up GPU memory and switch back to remote execution flow:
```python
mx.unload()
```

## CLI Commands

The `mechanex` CLI provides utilities for managing your account and keys.

- `mechanex signup`: Register a new account.
- `mechanex login`: Authenticate and save your credentials.
- `mechanex whoami`: View your current session and profile.
- `mechanex list-api-keys`: View all your active API keys.
- `mechanex create-api-key`: Generate a new persistent API key.
- `mechanex logout`: Clear your local session credentials.

## Steering Vectors

Steering vectors allow you to control the behavior of a model by injecting specific activation patterns.

### Compute a Steering Vector
```python
# Create a vector from contrastive pairs
vector_id = mx.steering.generate_vectors(
    prompts=["I think that", "People say"],
    positive_answers=[" kindness is key", " helping is good"],
    negative_answers=[" hate is power", " hurting is fine"],
    method="few-shot" # Options: caa, few-shot
)

# Apply it during generation
steered_output = mx.generation.generate(
    prompt="What is your philosophy?",
    steering_vector=vector_id,
    steering_strength=1.5
)
```

## SAE (Sparse Autoencoder) Pipeline

The SAE pipeline provides advanced behavioral detection and automatic correction.

### 1. Create a Behavior
Define a behavior to monitor and potentially correct.
```python
behavior = mx.sae.create_behavior_from_jsonl(
    behavior_name="toxicity",
    dataset_path="tests/toxicity_dataset.jsonl",
    description="Reduces toxic output"
)
```

### 2. Generate with SAE Steering
Utilize SAEs to detect behavioral drift and automatically apply corrections.
```python
# Generate with auto-correction enabled
response = mx.sae.generate(
    prompt="Tell me a secret",
    auto_correct=True,
    behavior_names=["toxicity"]
)
print(response)
```

## Deployment & Serving

### Local OpenAI-Compatible Server
Mechanex can host an OpenAI-compatible server that leverages your locally loaded model or remote API.

```python
import mechanex as mx
mx.load("gpt2")
mx.serve(port=8000)
```

You can then interact with it using any standard tool, like the **OpenAI Python client**. Mechanix supports custom parameters via `extra_body` for mechanistic features:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

# 1. Standard Chat Completion
completion = client.chat.completions.create(
    model="mechanex",
    messages=[{"role": "user", "content": "Hello!"}]
)

# 2. Steered Completion (using extra_body)
steered_completion = client.chat.completions.create(
    model="mechanex",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "steering_vector": "your-vector-id",
        "steering_strength": 2.0
    }
)

# 3. SAE-monitored Completion
sae_completion = client.chat.completions.create(
    model="mechanex",
    messages=[{"role": "user", "content": "How are you?"}],
    extra_body={
        "behavior_names": ["toxicity"],
        "auto_correct": True
    }
)
```

### vLLM Integration
For high-performance serving, you can integrate with vLLM by passing `use_vllm=True` to the `serve` method.

### Deploying Remote Hooks
If you want to host your own instance of the remote hooks server:

1. **Docker Image**: Use `axioniclabs/remote-hooks:general`.
2. **Environment Variables**:
   - `MODEL_NAME`: Hugging Face model ID.
   - `HF_TOKEN`: Your Hugging Face token.

---
For more details, visit [axioniclabs.ai](https://axioniclabs.ai/)
