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

# Load a model
mx.load_model("meta-llama/Llama-3.1-8B")
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

---
For more details, visit [axioniclabs.ai](https://axioniclabs.ai/)
