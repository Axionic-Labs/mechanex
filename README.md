# Mechanex

Mechanex is a runtime control layer for small models.

The core promise is:
**improve model behavior at inference time through policies, without retraining.**

## Product Direction

Mechanex is built around a policy-first workflow:

1. Choose a model (local, self-hosted, or hosted).
2. Choose a task profile.
3. Choose an objective.
4. Apply runtime controls (sampling, steering, constraints, verifiers, optimization).
5. Compare and evaluate policies.
6. Deploy and iterate.

## Core Concepts

### Policy
A policy is the reusable runtime object. It defines:
- Sampling method and search settings.
- Steering settings.
- Output constraints.
- Verifier stack.
- Optimization and fallback behavior.

### Execution Modes
Mechanex supports hybrid execution:
- `auto`: remote when authenticated, local when not authenticated.
- `remote`: force hosted inference (account + credits required).
- `local`: force local model execution.

## Hosted Remote Model Catalog

When using hosted execution (`mx.set_execution_mode("remote")`), you can select a model with:

```python
import mechanex as mx
mx.set_model("qwen3-0.6b")
```

Supported hosted models:

| Family | Models |
| :--- | :--- |
| **Gemma 2** | `gemma-2-27b`, `gemma-2-2b`, `gemma-2-9b`, `gemma-2-9b-it`, `gemma-2b`, `gemma-2b-it` |
| **Gemma 3** | `gemma-3-12b-it`, `gemma-3-12b-pt`, `gemma-3-1b-it`, `gemma-3-1b-pt`, `gemma-3-270m`, `gemma-3-270m-it`, `gemma-3-27b-it`, `gemma-3-27b-pt`, `gemma-3-4b-it`, `gemma-3-4b-pt` |
| **Llama** | `llama-3.1-8b`, `llama-3.1-8b-instruct`, `llama-3.3-70b-instruct`, `meta-llama-3-8b-instruct` |
| **Qwen** | `qwen2.5-7b-instruct`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-14b`, `qwen3-4b`, `qwen3-8b` |
| **Other** | `deepseek-r1-distill-llama-8b`, `gpt-oss-20b`, `gpt2-small`, `mistral-7b`, `pythia-70m-deduped` |

## Installation

```bash
pip install mechanex
```

## Quickstart

### Remote Runtime Policy

```python
import mechanex as mx

mx.set_key("your-api-key", persist=True)
mx.set_execution_mode("remote")

schema = {
    "type": "object",
    "required": ["answer", "confidence"],
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    }
}

out = mx.generation.generate(
    prompt="Return JSON: answer + confidence for 'Paris is in France'.",
    sampling_method="guided-generation",
    json_schema=schema,
    max_tokens=120,
)
print(out)
```

### Local Runtime Policy

```python
import mechanex as mx

mx.load("gpt2-small")
mx.set_execution_mode("local")

policy = mx.policy.strict_json_extraction(
    schema={
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    },
    name="strict_json_small_v1",
)

res = mx.policy.run(
    prompt="Summarize speculative decoding in one sentence.",
    policy=policy,
    include_trace=True,
)
print(res["output"])
```

## Runtime Controls

### Sampling Methods

Supported methods:
- `greedy`
- `top-k`
- `top-p`
- `min-p`
- `typical`
- `ads` (Adaptive Determinantal Sampling)
- `constrained-beam-search`
- `speculative-decoding`
- `ssd`
- `guided-generation`
- `ensemble-sampling`

### Steering Modes

Supported vector-generation modes:
- `caa`
- `few-shot`
- 'perceptrons'

### Constraints and Verifiers

Mechanex policies support:
- JSON mode + JSON schema constraints.
- Regex and grammar constraints.
- Required fields and forbidden terms.
- Verifier pipelines including syntax/schema checks.

## Local vs Remote Capability Notes

- `ADS` is remote-only.
- Steering perceptrons are remote-only.
- Other runtime policy methods are available for local usage with capability-aware fallback behavior where needed.

## Policy API

### Save, Run, Compare, Evaluate

```python
import mechanex as mx

policy = mx.policy.fast_tool_router()
pid = mx.policy.save(policy)

single = mx.policy.run(
    prompt="Route this request to the correct tool and return JSON.",
    policy_id=pid,
    include_trace=True,
)

cmp = mx.policy.compare(
    prompt="Extract order_id and status from text.",
    policies=[mx.policy.fast_tool_router(), mx.policy.strict_json_extraction({
        "type": "object",
        "required": ["order_id", "status"],
        "properties": {"order_id": {"type": "string"}, "status": {"type": "string"}},
    })],
)

ev = mx.policy.evaluate(
    prompts=[
        "Extract {name, role} from: Alice is CTO.",
        "Extract {name, role} from: Bob is PM.",
    ],
    policy_id=pid,
)
```

## OpenAI-Compatible Serving

Mechanex can run a local OpenAI-compatible server:

```python
import mechanex as mx
mx.load("gpt2-small")
mx.set_execution_mode("local")
mx.serve(port=8001)
```

Then use any OpenAI-compatible client against `http://localhost:8001/v1`.
`policy` / `policy_id`, steering fields, and behavior fields can be passed in request bodies.

## SDK Configuration

- Default backend URL is hosted Axionic backend.
- Override backend URL with:
  - constructor: `Mechanex(base_url="...")`
  - env var: `MECHANEX_BASE_URL`

## CLI

Account and key lifecycle:
- `mechanex signup`
- `mechanex login`
- `mechanex whoami`
- `mechanex create-api-key`
- `mechanex list-api-keys`
- `mechanex balance`
- `mechanex topup`
- `mechanex logout`

## Examples

See [examples/README.md](examples/README.md) for runnable workflows, including:
- local-first runtime control
- remote runtime control
- sampling strategy sweep
- strict JSON policies
- policy compare/evaluate
- local steering vectors
- OpenAI-compatible serving

## Engineering Docs

- [Contributing](CONTRIBUTING.md)
- [Engineering Standards](docs/ENGINEERING_STANDARDS.md)
- [Testing Guide](docs/TESTING.md)
- [CI/CD](docs/CI_CD.md)
- [Operations Runbook](docs/OPERATIONS_RUNBOOK.md)
- [Release Process](docs/RELEASE_PROCESS.md)
