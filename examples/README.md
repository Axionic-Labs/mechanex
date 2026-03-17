# Mechanex Examples and Use Cases

This folder contains runnable examples for local-first and remote policy runtime workflows.

## Prerequisites

- Python environment with `mechanex` installed.
- For remote examples:
  - `MECHANEX_BASE_URL` (optional override; defaults to hosted Axionic backend)
  - `MECHANEX_API_KEY`
- For local examples:
  - A local model loaded via `mx.load_model(...)` (requires `transformer-lens`).

## Example Catalog

1. `01_local_first_quickstart.py`
Local policy run with strict JSON constraints and trace output.

2. `02_remote_quickstart.py`
Remote generation with authentication and runtime controls.

3. `03_sampling_strategies.py`
Sampling strategy sweep across top-k/top-p/min-p/typical/speculative/guided/ensemble/ADS.

4. `04_strict_json_policy.py`
Policy preset for strict JSON extraction with schema verification.

5. `05_policy_compare_and_evaluate.py`
Policy comparison and evaluation sweep over a prompt suite.

6. `06_local_steering_vectors.py`
Generate local steering vectors and apply them during generation.

7. `07_openai_compatible_server.py`
Run an OpenAI-compatible server from Mechanex with policy support.

8. `08_hybrid_local_remote_toggle.py`
Switch runtime modes (`auto`, `local`, `remote`) in one script.

9. `09_sae_behavior_workflow.py`
Create a local SAE behavior and generate with behavior correction.

10. `10_remote_policy_smoke.py`
Remote policy save + run smoke flow.

## Run

```bash
python examples/01_local_first_quickstart.py
python examples/02_remote_quickstart.py
```
