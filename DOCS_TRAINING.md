# Mechanex Training Guide

This guide demonstrates how to use the `TrainingModule` in Mechanex for end-to-end model refinement—from data generation to deployment.

## Initializing the Client

First, ensure you have your API key set. The training features are powered by the remote Axionic API.

```python
import mechanex as mx

mx.set_key("your-api-key")
```

---

## 1. Data Generation

Generate high-quality seed data for training using a teacher model (e.g., Gemini).

```python
# Generate 50 seeds for a CRM application
task_id = mx.training.generate_data(
    num_seeds=50,
    output_file="crm_seeds.jsonl",
    topic="customer relationship management",
    teacher_provider="google",
    teacher_model="gemini-2.0-flash"
)

print(f"Data generation task queued: {task_id}")
```

---

## 2. Supervised Fine-Tuning (SFT)

The first step in RLHF is usually SFT, where we fine-tune a base model on high-quality demonstrations.

```python
# Fine-tune Gemma-2b on our generated seeds
task_id = mx.training.train_sft(
    base_model="google/gemma-2b-it",
    prompts_file="crm_seeds.jsonl",
    output_dir="crm_sft_model",
    epochs=3,
    use_peft=True
)

print(f"SFT training task queued: {task_id}")
```

---

## 3. Reinforcement Learning (RL)

Once you have an SFT model, you can further refine it using RL.

```python
# Run RL training on the SFT checkpoint
task_id = mx.training.train_rl(
    model_name_or_path="crm_sft_model",
    output_dir="crm_rl_model",
    prompts_file="rl_prompts.jsonl",
    num_train_epochs=1,
    batch_size=8
)

print(f"RL training task queued: {task_id}")
```

---

## 4. Evaluation

Evaluate your trained model to see how it performs on specific benchmarks or test sets.

```python
# Run evaluation on the RL model
task_id = mx.training.run_eval(
    model_name="crm_rl_model",
    output_dir="eval_results",
    num_eval_samples=100
)

print(f"Evaluation task queued: {task_id}")
```

---

## 5. Deployment

Deploy your final model to a vLLM-powered inference endpoint.

```python
# Deploy the RL model
task_id = mx.training.deploy(
    model_path="crm_rl_model",
    extra_args=["--port", "8000", "--gpu-memory-utilization", "0.9"]
)

print(f"Deployment task queued: {task_id}")
```

---

## Internal Architecture Note

These APIs act as a high-level wrapper around the Axionic Backend. Requests are proxied to a dedicated RL Pipeline service.

### Tenant Isolation
All file paths (`output_dir`, `prompts_file`, etc.) are automatically namespaced to your user account to ensure data isolation. If you provide `my_model`, it is stored internally as `user_<id>/my_model`.

---

## Next Steps
- Check out the [Getting Started Notebook](file:///Users/anugyandas/Axionic/mechanex/getting-started.ipynb) for local model steering.
- See the [Serving Demo](file:///Users/anugyandas/Axionic/mechanex/serving-demo.ipynb) for hosting your own OpenAI-compatible server.
