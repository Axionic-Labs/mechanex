# Mechanex

Mechanex is a powerful Python library for interacting with the Axionic API, designed for mechanistically debugging, steering, and monitoring Large Language Models (LLMs).

Learn more at [axioniclabs.ai](https://axioniclabs.ai/)

## Installation

```bash
pip install mechanex
```

## Quick Start

### 1. Initialize the Client

```python
import mechanex as mx
import os

# Set your API key
mx.set_key("your-api-key-here")
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

### Load from JSONL
You can also generate vectors from a dataset:
```python
vector_id = mx.steering.generate_from_jsonl("path/to/dataset.jsonl")
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
    behavior_names=["toxicity"] # Optional: filter for specific behaviors
)
print(response) # Returns the generated text string
```

### 3. List Behaviors
```python
all_behaviors = mx.sae.list_behaviors()
```

## Model Utilities

### Inspect the Model Graph
Return the layer structure of the currently loaded model.
```python
graph = mx.model.get_graph()
for node in graph:
    print(node)
```

## Deploying Remote Hooks

If you want to host your own instance of the remote hooks server, follow these steps:

1. **Model Hosting**: Ensure the model you want to use is available on [Hugging Face](https://huggingface.co/).
2. **Docker Image**: Use the official Docker image: `axioniclabs/remote-hooks:general`. You can find it on [Docker Hub](https://hub.docker.com/r/axioniclabs/remote-hooks).
3. **Configuration**: Set the following environment variables in your deployment:
   - `MODEL_NAME`: The Hugging Face model identifier (e.g., `meta-llama/Llama-3.2-1B`).
   - `HF_TOKEN`: Your Hugging Face access token.

---
For more details, visit [axioniclabs.ai](https://axioniclabs.ai/)
