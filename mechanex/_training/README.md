# Mechanex RL Pipeline: Engineering Plan & Roadmap

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Component Specifications](#3-component-specifications)
4. [Data Engineering Pipeline](#4-data-engineering-pipeline)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Risk Assessment & Mitigations](#8-risk-assessment--mitigations)
9. [Technical References](#9-technical-references)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### 1.1 Vision

The Mechanex RL Pipeline enables enterprises to create lightweight (<1B parameter), task-specific tool-calling agents that achieve frontier-model accuracy at a fraction of the cost and latency. The system automates the traditionally manual processes of reward engineering and training data curation through a novel Teacher → Assistant → Student distillation hierarchy.

### 1.2 Core Value Propositions

| Metric | Frontier Model | Mechanex Student | Improvement |
|--------|---------------|------------------|-------------|
| Inference Latency | 2-5 seconds | 50-200ms | 10-25x faster |
| Cost per 1M calls | $15-60 | $0.50-2.00 | 30-50x cheaper |
| VRAM Requirement | 80GB+ | 2-4GB | 20-40x smaller |
| JSON Schema Compliance | 95-98% | Target: 97%+ | Parity |

### 1.3 Key Innovations

1. **Automated Reward Architect (ARA):** Compiler-driven reward function generation eliminating manual reward engineering
2. **Three-Stage Distillation:** Teacher → Assistant → Student topology that preserves reasoning density across capacity gaps
3. **Dr.GRPO Algorithm:** Memory-efficient RL that computes relative rewards within sample groups, eliminating critic networks

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MECHANEX RL PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   TEACHER    │    │  ASSISTANT   │    │   STUDENT    │                   │
│  │  (Frontier)  │───▶│   (7B-14B)   │───▶│  (<1B/Mini)  │                   │
│  │              │    │              │    │              │                   │
│  │ • GPT-4/4.1  │    │ • Qwen-2.5-7B│    │ • Llama-3.3-1B│                  │
│  │ • Claude 3.5 │    │ • Llama-4-8B │    │ • Mechanex-Mini│                 │
│  │ • Gemini 2.0 │    │              │    │   (250-500M) │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────▲───────┘                   │
│         │                   │                   │                           │
│         │ Golden Traces     │ Compressed        │ Optimized                 │
│         │ + Reward Code     │ Trajectories      │ Policy                    │
│         ▼                   ▼                   │                           │
│  ┌─────────────────────────────────────────────┴──────┐                     │
│  │              AUTOMATED REWARD ARCHITECT (ARA)      │                     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │                     │
│  │  │  Format     │  │ Constraint  │  │   Type     │ │                     │
│  │  │  Validator  │  │  Grounder   │  │  Checker   │ │                     │
│  │  └─────────────┘  └─────────────┘  └────────────┘ │                     │
│  └───────────────────────────┬───────────────────────┘                     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────┐                     │
│  │              Dr.GRPO TRAINING ENGINE              │                     │
│  │  • Group-relative reward computation              │                     │
│  │  • KL-divergence regularization (vs Assistant)   │                     │
│  │  • Online trajectory sampling                     │                     │
│  └───────────────────────────────────────────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagram

```
┌─────────────────┐
│  Tool Schema    │──────┐
│  (JSON/NL)      │      │
└─────────────────┘      │
                         ▼
┌─────────────────┐    ┌─────────────────────────────────┐
│  Enterprise     │───▶│  PHASE 1: Seed Generation       │
│  Context Docs   │    │  • Teacher generates 5K prompts │
└─────────────────┘    │  • ARA compiles reward function │
                       └───────────────┬─────────────────┘
                                       │
                                       ▼
                       ┌─────────────────────────────────┐
                       │  PHASE 2: Trajectory Expansion  │
                       │  • Assistant generates 8x       │
                       │    trajectories per prompt      │
                       │  • 4 correct + 4 error-recovery │
                       └───────────────┬─────────────────┘
                                       │
                                       ▼
                       ┌─────────────────────────────────┐
                       │  PHASE 3: Quality Filtering     │
                       │  • Teacher verification         │
                       │  • ARA syntactic validation     │
                       │  • Diversity deduplication      │
                       └───────────────┬─────────────────┘
                                       │
                                       ▼
                       ┌─────────────────────────────────┐
                       │  PHASE 4: Cold Start SFT        │
                       │  • 50K filtered traces          │
                       │  • 2 epochs, establish baseline │
                       └───────────────┬─────────────────┘
                                       │
                                       ▼
                       ┌─────────────────────────────────┐
                       │  PHASE 5: Dr.GRPO Alignment     │
                       │  • Online RL with ARA rewards   │
                       │  • KL constraint vs Assistant   │
                       └───────────────┬─────────────────┘
                                       │
                                       ▼
                       ┌─────────────────────────────────┐
                       │  PHASE 6: Specialist Model      │
                       │  • Exported for deployment      │
                       │  • Quantized (INT8/INT4)        │
                       └─────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Automated Reward Architect (ARA)

#### 3.1.1 Purpose

The ARA module treats tool definitions as formal specifications and the Teacher model as a compiler, automatically generating executable Python reward functions from natural language descriptions.

#### 3.1.2 Architecture

```python
# ARA Module Structure
class ARAModule:
    """
    Automated Reward Architect - Compiles tool specs into reward functions.
    """

    def __init__(self, teacher_client: TeacherClient, config: ARAConfig):
        self.teacher = teacher_client
        self.config = config
        self.reward_cache = {}

    def compile(self, tool_schema: ToolSchema) -> GeneratedReward:
        """
        Input: JSON/NL tool schema
        Output: Executable GeneratedReward class
        """
        # 1. Parse tool schema into canonical form
        canonical = self._canonicalize_schema(tool_schema)

        # 2. Generate reward components via Teacher
        reward_code = self._teacher_compile(canonical)

        # 3. Validate generated code in sandbox
        validated = self._sandbox_validate(reward_code)

        # 4. Return executable reward class
        return self._instantiate_reward(validated)
```

#### 3.1.3 Reward Function Components

The ARA generates a `MXReward` class with three core validators:

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Format Validator** | Ensures output contains required structural elements | Regex matching for `<think>`, `<tool_call>` tags |
| **Constraint Grounder** | Validates arguments against prompt context | Semantic matching + entity extraction |
| **Type Checker** | Validates JSON body against schema | Pydantic model validation |

#### 3.1.4 Generated Reward Class Template

```python
class MXReward:
    """Auto-generated by ARA for tool: {tool_name}"""

    def __init__(self, tool_schema: dict, context_entities: List[str]):
        self.schema = tool_schema
        self.context_entities = context_entities
        self.pydantic_model = self._build_validator()

    def __call__(self, response: str, prompt: str) -> RewardResult:
        """
        Compute composite reward for a model response.

        Returns:
            RewardResult with scores and detailed breakdown
        """
        scores = {}

        # 1. Format Check (0.0 - 1.0)
        scores['format'] = self._check_format(response)

        # 2. Grounding Check (0.0 - 1.0)
        scores['grounding'] = self._check_grounding(response, prompt)

        # 3. Type Check (0.0 - 1.0)
        scores['type_valid'] = self._check_types(response)

        # 4. Reasoning Quality (0.0 - 1.0) [IMPROVEMENT]
        scores['reasoning'] = self._check_reasoning_quality(response)

        # Composite score with configurable weights
        final_score = self._weighted_composite(scores)

        return RewardResult(
            score=final_score,
            breakdown=scores,
            passed=final_score >= self.config.pass_threshold
        )

    def _check_format(self, response: str) -> float:
        """Validate structural format compliance."""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)

        if not think_match or not tool_match:
            return 0.0

        # Partial credit for malformed but present tags
        score = 1.0
        if len(think_match.group(1).strip()) < 10:
            score -= 0.3  # Penalize empty reasoning

        return score

    def _check_grounding(self, response: str, prompt: str) -> float:
        """Validate arguments are grounded in prompt context."""
        tool_call = self._extract_tool_call(response)
        if not tool_call:
            return 0.0

        # Extract entities from prompt
        prompt_entities = self._extract_entities(prompt)

        # Check each argument value is grounded
        grounded_count = 0
        total_args = len(tool_call.get('arguments', {}))

        for arg_name, arg_value in tool_call.get('arguments', {}).items():
            if self._is_grounded(arg_value, prompt_entities, prompt):
                grounded_count += 1

        return grounded_count / max(total_args, 1)

    def _check_types(self, response: str) -> float:
        """Validate JSON body against Pydantic schema."""
        try:
            tool_call = self._extract_tool_call(response)
            self.pydantic_model(**tool_call.get('arguments', {}))
            return 1.0
        except ValidationError as e:
            # Partial credit based on error count
            error_count = len(e.errors())
            total_fields = len(self.pydantic_model.__fields__)
            return max(0, (total_fields - error_count) / total_fields)
```

#### 3.1.5 Improvements to Original Design

**IMPROVEMENT 1: Reasoning Quality Score**

Add a fourth reward component that evaluates the quality of the `<think>` block:

```python
def _check_reasoning_quality(self, response: str) -> float:
    """
    Evaluate reasoning trace quality.

    Criteria:
    - Mentions relevant tool parameters
    - Shows logical progression
    - References prompt context
    """
    think_block = self._extract_think_block(response)
    if not think_block:
        return 0.0

    score = 0.0

    # Check for parameter mentions
    param_names = list(self.schema.get('parameters', {}).keys())
    mentioned = sum(1 for p in param_names if p.lower() in think_block.lower())
    score += 0.3 * (mentioned / max(len(param_names), 1))

    # Check for logical connectors (indicates reasoning)
    logical_terms = ['because', 'therefore', 'since', 'so', 'need to', 'should']
    has_logic = any(term in think_block.lower() for term in logical_terms)
    score += 0.3 if has_logic else 0.0

    # Check minimum reasoning length
    word_count = len(think_block.split())
    score += 0.4 * min(word_count / 50, 1.0)  # Target ~50 words

    return score
```

**IMPROVEMENT 2: Hierarchical Reward Weights**

Different tasks should weight reward components differently:

```python
REWARD_WEIGHTS = {
    'json_extraction': {
        'format': 0.2,
        'grounding': 0.3,
        'type_valid': 0.4,
        'reasoning': 0.1
    },
    'tool_routing': {
        'format': 0.15,
        'grounding': 0.35,
        'type_valid': 0.2,
        'reasoning': 0.3
    },
    'sql_generation': {
        'format': 0.1,
        'grounding': 0.4,
        'type_valid': 0.3,
        'reasoning': 0.2
    }
}
```

**IMPROVEMENT 3: Reward Sandboxing**

Execute generated reward code in a secure sandbox to prevent code injection:

```python
class RewardSandbox:
    """Secure execution environment for generated reward code."""

    ALLOWED_IMPORTS = {'re', 'json', 'pydantic', 'typing'}
    MAX_EXECUTION_TIME = 5.0  # seconds

    def execute(self, reward_code: str, response: str, prompt: str) -> RewardResult:
        # Parse and validate AST
        tree = ast.parse(reward_code)
        self._validate_ast(tree)

        # Execute in restricted namespace
        namespace = self._create_restricted_namespace()

        with timeout(self.MAX_EXECUTION_TIME):
            exec(compile(tree, '<reward>', 'exec'), namespace)
            reward_class = namespace['MXReward']
            return reward_class()(response, prompt)
```

---

### 3.2 Model Topology Specifications

#### 3.2.1 Teacher Model (Frontier)

| Attribute | Specification |
|-----------|---------------|
| **Primary Options** | GPT-5.2, Claude 4.5 Sonnet/Opus, Gemini 2.5 Flash |
| **Role** | Golden trace generation, reward code compilation, trajectory verification |
| **API Integration** | OpenAI, Anthropic, Google AI Studio compatible |
| **Fallback Strategy** | Implement retry with exponential backoff, model rotation on rate limits |

**Teacher Prompt Template for Trace Generation:**

```
You are generating training data for a specialized tool-calling model.

TOOL DEFINITION:
{tool_schema}

USER CONTEXT:
{enterprise_context}

TASK:
Generate a high-quality reasoning trace that:
1. Analyzes the user's request in <think> tags
2. Produces a valid tool call in <tool_call> tags

CONSTRAINTS:
- All argument values MUST be grounded in the user context
- The <think> block should be 50-150 words explaining your reasoning
- The tool call must be valid JSON matching the schema

OUTPUT FORMAT:
<think>
[Your reasoning here]
</think>
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>
```

#### 3.2.2 Assistant Model (7B-14B)

| Attribute | Specification |
|-----------|---------------|
| **Primary Options** | Qwen-2.5-7B-Instruct, Llama-4-Scout-8B |
| **Role** | Reasoning compression, trajectory diversification, KL reference |
| **Deployment** | Self-hosted on 1-2x A100/H100 or cloud inference |
| **Fine-tuning** | Optional domain adaptation before pipeline use |

**Key Responsibilities:**

1. **Trajectory Expansion:** Generate 8 diverse trajectories per seed prompt
2. **Reasoning Compression:** Summarize Teacher's verbose reasoning into student-digestible format
3. **KL Reference:** Serve as the reference distribution for KL-divergence regularization during Dr.GRPO

**IMPROVEMENT: Diversity Prompting Strategy**

To ensure trajectory diversity, use structured variation prompts:

```python
DIVERSITY_PROMPTS = [
    # Correct executions (4 variants)
    "Solve this directly with minimal reasoning.",
    "Think step-by-step before generating the tool call.",
    "Consider edge cases, then generate the most robust call.",
    "Verify your assumptions about the data before calling.",

    # Error-recovery traces (4 variants)
    "Initially make a wrong assumption, realize it, then correct.",
    "First generate an invalid call, catch the error, then fix.",
    "Start with incomplete parameters, notice, then complete.",
    "Misinterpret a field initially, recognize the mistake, correct."
]
```

#### 3.2.3 Student Model (<1B)

| Attribute | Specification |
|-----------|---------------|
| **Initial Base** | Llama-3.3-1B-Instruct |
| **Target Base** | Mechanex-Mini (250-500M, custom architecture) |
| **Training** | SFT cold start → Dr.GRPO alignment |
| **Deployment** | Edge devices, CPU inference, quantized (INT4/INT8) |

**Student Output Format:**

```
<think>
I need to update the lead status for John Smith to "Qualified".
The CRM tool requires lead_id and new_status parameters.
From the context, John Smith's lead_id is "LD-2847".
</think>
<tool_call>
{"name": "update_lead_status", "arguments": {"lead_id": "LD-2847", "status": "Qualified"}}
</tool_call>
```

---

### 3.3 Dr.GRPO Training Engine

#### 3.3.1 Algorithm Overview

Dr.GRPO (Deep Reasoning Group Relative Policy Optimization) is a critic-free RL algorithm that computes rewards by comparing multiple sampled trajectories within a group, rather than requiring a separate value network.

**Key Advantages:**

1. **Memory Efficiency:** No critic network reduces VRAM by ~40%
2. **Stability:** Group-relative rewards reduce variance
3. **Reasoning Preservation:** KL constraint against Assistant maintains reasoning structure

#### 3.3.2 Mathematical Formulation

**Objective Function:**

```
L_DrGRPO(θ) = E_prompt[E_group[Σᵢ Aᵢ · log π_θ(yᵢ|x)]] - β · KL(π_θ || π_ref)
```

Where:
- `θ`: Student model parameters
- `x`: Input prompt
- `yᵢ`: i-th sampled trajectory in the group
- `Aᵢ`: Group-relative advantage for trajectory i
- `π_ref`: Assistant model (reference distribution)
- `β`: KL penalty coefficient (typically 0.01-0.1)

**Group-Relative Advantage Computation:**

```
Aᵢ = (R(yᵢ) - μ_group) / (σ_group + ε)
```

Where:
- `R(yᵢ)`: ARA reward for trajectory i
- `μ_group`: Mean reward in the sample group
- `σ_group`: Standard deviation of rewards in group
- `ε`: Small constant for numerical stability (1e-8)

#### 3.3.3 Implementation Specification

```python
class DrGRPOTrainer:
    """
    Deep Reasoning GRPO Trainer for student model optimization.

    Reference: Adapted from DeepSeek-R1 GRPO with reasoning-specific modifications
    """

    def __init__(
        self,
        student_model: PreTrainedModel,
        reference_model: PreTrainedModel,  # Assistant (frozen)
        reward_fn: MXReward,
        config: DrGRPOConfig
    ):
        self.student = student_model
        self.reference = reference_model
        self.reward_fn = reward_fn
        self.config = config

        # Freeze reference model
        for param in self.reference.parameters():
            param.requires_grad = False

    def compute_group_advantages(
        self,
        trajectories: List[str],
        prompt: str
    ) -> torch.Tensor:
        """
        Compute group-relative advantages for a batch of trajectories.
        """
        # Get rewards from ARA
        rewards = torch.tensor([
            self.reward_fn(traj, prompt).score
            for traj in trajectories
        ])

        # Normalize within group
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8

        advantages = (rewards - mean_reward) / std_reward

        return advantages

    def compute_kl_penalty(
        self,
        student_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty against reference model.
        """
        kl = (torch.exp(student_logprobs) *
              (student_logprobs - reference_logprobs)).sum(-1)
        return kl.mean()

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step of Dr.GRPO.
        """
        prompts = batch['prompts']

        total_loss = 0.0
        metrics = {'reward_mean': 0.0, 'kl': 0.0, 'advantage_std': 0.0}

        for prompt in prompts:
            # Sample G trajectories from student
            trajectories = self._sample_trajectories(
                prompt,
                num_samples=self.config.group_size
            )

            # Compute group advantages
            advantages = self.compute_group_advantages(trajectories, prompt)

            # Get log probabilities
            student_logprobs = self._get_logprobs(self.student, trajectories, prompt)
            ref_logprobs = self._get_logprobs(self.reference, trajectories, prompt)

            # Policy gradient loss
            pg_loss = -(advantages * student_logprobs).mean()

            # KL penalty
            kl_penalty = self.compute_kl_penalty(student_logprobs, ref_logprobs)

            # Combined loss
            loss = pg_loss + self.config.kl_coef * kl_penalty
            total_loss += loss

            # Track metrics
            metrics['reward_mean'] += advantages.mean().item()
            metrics['kl'] += kl_penalty.item()
            metrics['advantage_std'] += advantages.std().item()

        # Backward pass
        total_loss.backward()

        return metrics

    def _sample_trajectories(
        self,
        prompt: str,
        num_samples: int
    ) -> List[str]:
        """
        Sample diverse trajectories from current student policy.

        IMPROVEMENT: Early stopping for efficiency
        """
        trajectories = []

        with torch.no_grad():
            for _ in range(num_samples):
                output = self.student.generate(
                    prompt,
                    max_new_tokens=self.config.max_response_tokens,
                    temperature=self.config.sampling_temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    # Early stopping on </tool_call>
                    stop_strings=['</tool_call>'],
                )
                trajectories.append(output)

        return trajectories
```

#### 3.3.4 Hyperparameter Recommendations

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `group_size` | 8 | Trajectories per prompt |
| `kl_coef` (β) | 0.05 | Balance exploration vs reference fidelity |
| `learning_rate` | 1e-6 to 5e-6 | Conservative for stability |
| `batch_size` | 4-8 prompts | Depends on VRAM |
| `sampling_temperature` | 0.8 | Encourage diversity |
| `max_response_tokens` | 512 | Sufficient for think + tool_call |
| `gradient_accumulation` | 4-8 | Effective batch size scaling |
| `warmup_steps` | 100 | Before full KL penalty |

#### 3.3.5 Improvements to Original Dr.GRPO Design

**IMPROVEMENT 1: Curriculum Learning Schedule**

Start with easier prompts and progressively increase difficulty:

```python
class CurriculumScheduler:
    """
    Schedule training data by difficulty.
    """

    def __init__(self, dataset: Dataset, num_stages: int = 3):
        # Score prompts by complexity
        self.scored_data = self._score_by_difficulty(dataset)
        self.stages = self._partition_stages(num_stages)
        self.current_stage = 0

    def _score_by_difficulty(self, dataset: Dataset) -> List[Tuple[dict, float]]:
        """
        Difficulty heuristics:
        - Number of required parameters
        - Nested JSON depth
        - Context length
        - Semantic ambiguity (via Teacher scoring)
        """
        scored = []
        for item in dataset:
            score = (
                0.3 * len(item['tool_schema'].get('required', [])) / 10 +
                0.2 * self._json_depth(item['tool_schema']) / 5 +
                0.2 * len(item['context']) / 2000 +
                0.3 * item.get('ambiguity_score', 0.5)
            )
            scored.append((item, score))
        return sorted(scored, key=lambda x: x[1])

    def get_current_data(self) -> Dataset:
        """Return data for current curriculum stage."""
        return self.stages[self.current_stage]

    def advance_stage(self):
        """Move to next difficulty level."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
```

**IMPROVEMENT 2: Reward Shaping for Partial Credit**

Decompose rewards to provide learning signal even for failures:

```python
def shaped_reward(self, response: str, prompt: str) -> float:
    """
    Shaped reward that provides partial credit.
    """
    base_result = self.reward_fn(response, prompt)

    # If complete failure, check for partial progress
    if base_result.score < 0.3:
        partial_score = 0.0

        # Credit for attempting correct format
        if '<think>' in response:
            partial_score += 0.1
        if '<tool_call>' in response:
            partial_score += 0.1

        # Credit for mentioning correct tool
        tool_schema = self.reward_fn.schema
        if tool_schema['name'] in response:
            partial_score += 0.1

        return max(base_result.score, partial_score)

    return base_result.score
```

**IMPROVEMENT 3: Dynamic KL Coefficient**

Adapt KL penalty based on training progress:

```python
def adaptive_kl_coef(self, step: int, metrics: Dict[str, float]) -> float:
    """
    Adjust KL coefficient based on training dynamics.

    - Increase if KL diverges too much (student drifting)
    - Decrease if rewards plateau (allow more exploration)
    """
    base_kl = self.config.kl_coef
    current_kl = metrics['kl']
    reward_improvement = metrics.get('reward_delta', 0)

    if current_kl > self.config.kl_target * 1.5:
        # Drifting too far, increase penalty
        return min(base_kl * 1.5, 0.2)
    elif reward_improvement < 0.01 and step > 500:
        # Plateauing, allow more exploration
        return max(base_kl * 0.8, 0.01)

    return base_kl
```

---

## 4. Data Engineering Pipeline

### 4.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ENGINEERING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Seed Generation                                           │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ Tool Schema │────▶│  Teacher    │────▶│ 5K Golden   │           │
│  │ + Context   │     │  Generation │     │   Prompts   │           │
│  └─────────────┘     └─────────────┘     └──────┬──────┘           │
│                                                  │                   │
│  Stage 2: Trajectory Expansion                   ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ 5K Prompts  │────▶│  Assistant  │────▶│ 40K Raw     │           │
│  │             │     │  8x Expand  │     │ Trajectories│           │
│  └─────────────┘     └─────────────┘     └──────┬──────┘           │
│                                                  │                   │
│  Stage 3: Quality Filtering                      ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ 40K Raw     │────▶│ ARA + Teacher│───▶│ 50K Filtered│           │
│  │ Trajectories│     │ Validation   │     │   (SFT Set) │           │
│  └─────────────┘     └─────────────┘     └──────┬──────┘           │
│                                                  │                   │
│  Stage 4: Training Data Format                   ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ {"prompt": "...", "completion": "<think>...</think>...",    │   │
│  │  "reward_breakdown": {"format": 1.0, "grounding": 0.95...}} │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Stage 1: Seed Data Generation

#### 4.2.1 Prompt Template for Seed Generation

```python
SEED_GENERATION_PROMPT = """
You are generating diverse, realistic enterprise prompts for tool-calling training.

TOOL DEFINITION:
{tool_schema}

ENTERPRISE CONTEXT:
{context_description}

Generate {num_prompts} unique user prompts that would require using this tool.
Each prompt should:
1. Be realistic enterprise scenarios
2. Include specific entities (names, IDs, values) that ground the tool parameters
3. Vary in complexity (simple direct requests to nuanced multi-step needs)
4. Include edge cases ({edge_case_percentage}% should be boundary conditions)

OUTPUT FORMAT:
Return a JSON array of prompt objects:
[
  {
    "prompt": "User request text",
    "expected_entities": {"param1": "value1", ...},
    "complexity": "simple|medium|complex",
    "is_edge_case": false
  },
  ...
]
"""
```

#### 4.2.2 Seed Generation Configuration

```python
@dataclass
class SeedGenerationConfig:
    num_prompts: int = 5000
    edge_case_percentage: float = 0.15  # 15% edge cases
    complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        'simple': 0.3,
        'medium': 0.5,
        'complex': 0.2
    })
    teacher_temperature: float = 0.9  # Higher for diversity
    batch_size: int = 50  # Prompts per API call

    # Quality filters
    min_prompt_length: int = 20
    max_prompt_length: int = 500
    require_grounded_entities: bool = True
```

### 4.3 Stage 2: Trajectory Expansion

#### 4.3.1 Diversity-Constrained Generation

```python
class TrajectoryExpander:
    """
    Expands seed prompts into diverse trajectory sets.
    """

    def __init__(self, assistant_model: AssistantModel, config: ExpansionConfig):
        self.assistant = assistant_model
        self.config = config

    def expand(self, seed_prompt: str, tool_schema: dict) -> List[Trajectory]:
        """
        Generate 8 diverse trajectories per seed prompt.

        Distribution:
        - 4 correct executions (varying reasoning styles)
        - 4 error-recovery traces (mistake → correction)
        """
        trajectories = []

        # Generate correct executions
        for i in range(4):
            style = self.config.correct_styles[i]
            traj = self._generate_correct(seed_prompt, tool_schema, style)
            trajectories.append(traj)

        # Generate error-recovery traces
        for i in range(4):
            error_type = self.config.error_types[i]
            traj = self._generate_error_recovery(seed_prompt, tool_schema, error_type)
            trajectories.append(traj)

        return trajectories

    def _generate_correct(
        self,
        prompt: str,
        schema: dict,
        style: str
    ) -> Trajectory:
        """Generate a correct trajectory with specified reasoning style."""

        style_instructions = {
            'minimal': "Be concise. Reason briefly then act.",
            'step_by_step': "Think through each step methodically.",
            'edge_aware': "Consider what could go wrong, then proceed safely.",
            'verification': "Verify your assumptions before acting."
        }

        system_prompt = f"""
        {style_instructions[style]}

        Generate a response with <think> reasoning and <tool_call> execution.
        The tool call must be valid JSON matching this schema:
        {json.dumps(schema, indent=2)}
        """

        response = self.assistant.generate(
            system=system_prompt,
            user=prompt,
            temperature=0.7
        )

        return Trajectory(
            prompt=prompt,
            response=response,
            style=style,
            is_error_recovery=False
        )

    def _generate_error_recovery(
        self,
        prompt: str,
        schema: dict,
        error_type: str
    ) -> Trajectory:
        """Generate a trajectory demonstrating error recovery."""

        error_instructions = {
            'wrong_assumption': """
                First, make an incorrect assumption about the data.
                In your <think> block, realize the mistake.
                Then correct yourself and generate the right tool call.
            """,
            'invalid_json': """
                First, generate a malformed JSON in your thinking.
                Catch the syntax error in your <think> block.
                Then provide the corrected, valid tool call.
            """,
            'missing_param': """
                First, attempt a call with missing required parameters.
                Notice what's missing in your <think> block.
                Then generate the complete, correct call.
            """,
            'type_error': """
                First, use the wrong type for a parameter (e.g., string instead of number).
                Catch this in your <think> block.
                Then correct the type and generate proper call.
            """
        }

        system_prompt = f"""
        {error_instructions[error_type]}

        This teaches error-recovery behavior. Show your mistake-finding process.
        Final tool call must be valid JSON matching:
        {json.dumps(schema, indent=2)}
        """

        response = self.assistant.generate(
            system=system_prompt,
            user=prompt,
            temperature=0.8
        )

        return Trajectory(
            prompt=prompt,
            response=response,
            style='error_recovery',
            is_error_recovery=True,
            error_type=error_type
        )
```

### 4.4 Stage 3: Quality Filtering

#### 4.4.1 Multi-Stage Filter Pipeline

```python
class QualityFilter:
    """
    Multi-stage filtering pipeline for trajectory quality assurance.
    """

    def __init__(
        self,
        ara_module: ARAModule,
        teacher_client: TeacherClient,
        config: FilterConfig
    ):
        self.ara = ara_module
        self.teacher = teacher_client
        self.config = config

    def filter(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Apply filtering stages:
        1. ARA syntactic validation
        2. Teacher semantic verification
        3. Diversity deduplication
        4. Balance enforcement
        """

        # Stage 1: ARA Validation
        ara_passed = []
        for traj in trajectories:
            result = self.ara.validate(traj.response, traj.prompt)
            if result.score >= self.config.ara_threshold:
                traj.ara_score = result.score
                traj.ara_breakdown = result.breakdown
                ara_passed.append(traj)

        print(f"ARA Filter: {len(trajectories)} → {len(ara_passed)}")

        # Stage 2: Teacher Verification
        teacher_passed = []
        for traj in ara_passed:
            is_correct = self._teacher_verify(traj)
            if is_correct:
                teacher_passed.append(traj)

        print(f"Teacher Filter: {len(ara_passed)} → {len(teacher_passed)}")

        # Stage 3: Diversity Deduplication
        deduplicated = self._deduplicate(teacher_passed)
        print(f"Deduplication: {len(teacher_passed)} → {len(deduplicated)}")

        # Stage 4: Balance Enforcement
        balanced = self._enforce_balance(deduplicated)
        print(f"Balancing: {len(deduplicated)} → {len(balanced)}")

        return balanced

    def _teacher_verify(self, trajectory: Trajectory) -> bool:
        """
        Use Teacher model to verify semantic correctness.
        """
        verification_prompt = f"""
        Verify if this tool call is semantically correct for the given prompt.

        USER PROMPT:
        {trajectory.prompt}

        MODEL RESPONSE:
        {trajectory.response}

        TOOL SCHEMA:
        {trajectory.tool_schema}

        Answer with JSON: {{"correct": true/false, "reason": "..."}}
        """

        result = self.teacher.generate(verification_prompt, temperature=0.0)
        parsed = json.loads(result)

        return parsed.get('correct', False)

    def _deduplicate(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Remove near-duplicate trajectories using embedding similarity.
        """
        embeddings = self._compute_embeddings(trajectories)

        unique = []
        unique_embeddings = []

        for traj, emb in zip(trajectories, embeddings):
            is_duplicate = False
            for existing_emb in unique_embeddings:
                similarity = cosine_similarity(emb, existing_emb)
                if similarity > self.config.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(traj)
                unique_embeddings.append(emb)

        return unique

    def _enforce_balance(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Ensure balanced distribution across complexity levels and styles.
        """
        # Group by characteristics
        by_complexity = defaultdict(list)
        by_style = defaultdict(list)

        for traj in trajectories:
            by_complexity[traj.complexity].append(traj)
            by_style[traj.style].append(traj)

        # Sample to enforce balance
        target_per_complexity = self.config.target_size // 3
        target_per_style = self.config.target_size // 8

        balanced = []

        # ... sampling logic to achieve target distribution

        return balanced
```

### 4.5 Data Format Specification

#### 4.5.1 SFT Training Format

```json
{
  "id": "traj_001_v3",
  "prompt": "Update the status of lead John Smith (ID: LD-2847) to Qualified in our CRM.",
  "completion": "<think>\nI need to update the lead status in the CRM system.\n\nFrom the request:\n- Lead name: John Smith\n- Lead ID: LD-2847  \n- New status: Qualified\n\nThe update_lead_status tool requires lead_id and status parameters.\nI have both values from the context.\n</think>\n<tool_call>\n{\"name\": \"update_lead_status\", \"arguments\": {\"lead_id\": \"LD-2847\", \"status\": \"Qualified\"}}\n</tool_call>",
  "metadata": {
    "tool_name": "update_lead_status",
    "complexity": "simple",
    "style": "step_by_step",
    "is_error_recovery": false,
    "source": "assistant_expanded",
    "ara_score": 0.97,
    "teacher_verified": true
  }
}
```

#### 4.5.2 Dr.GRPO Training Format

```json
{
  "id": "grpo_batch_042",
  "prompt": "Find all invoices from Acme Corp over $10,000 from last quarter",
  "tool_schema": {
    "name": "search_invoices",
    "parameters": {
      "vendor": {"type": "string"},
      "min_amount": {"type": "number"},
      "date_range": {"type": "object"}
    }
  },
  "reference_completion": "...",  // From Assistant for KL computation
  "reward_function_id": "ara_search_invoices_v2"
}
```

---

## 5. Training Infrastructure

### 5.1 Hardware Requirements

#### 5.1.1 Development Environment

| Component | Specification | Purpose |
|-----------|---------------|---------|
| GPU | 1x RTX 4090 (24GB) or A100 (40GB) | Local development, small experiments |
| CPU | 16+ cores | Data preprocessing |
| RAM | 64GB+ | Dataset loading |
| Storage | 1TB NVMe SSD | Model checkpoints, datasets |

#### 5.1.2 Training Environment

| Phase | Hardware | Duration (Est.) | Cost (Cloud) |
|-------|----------|-----------------|--------------|
| SFT Cold Start | 4x A100 80GB | 8-16 hours | $200-400 |
| Dr.GRPO Alignment | 8x A100 80GB | 24-48 hours | $1,200-2,400 |
| Evaluation | 2x A100 80GB | 4-8 hours | $80-160 |

**Total estimated cloud cost per training run: $1,500 - $3,000**

#### 5.1.3 Inference Environment (Production)

| Deployment Target | Hardware | Throughput |
|-------------------|----------|------------|
| Cloud API | 1x A10G (24GB) | ~100 req/s |
| Edge Server | 1x RTX 4060 (8GB) | ~30 req/s |
| CPU Only | 8-core x86 | ~5 req/s |
| Quantized (INT4) | Any GPU 4GB+ | ~80 req/s |

### 5.2 Software Stack

```yaml
# environment.yaml
name: mechanex-rl
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.2
  - cuda=12.1

  # Training
  - pip:
    - transformers>=4.40
    - trl>=0.8  # Hugging Face RL library
    - peft>=0.10  # LoRA/QLoRA
    - bitsandbytes>=0.43  # Quantization
    - deepspeed>=0.14  # Distributed training
    - flash-attn>=2.5  # Efficient attention

    # Data
    - datasets>=2.18
    - pandas>=2.2
    - pyarrow>=15

    # Evaluation
    - lm-eval>=0.4  # LM Evaluation Harness
    - pytest>=8.0

    # API Clients
    - openai>=1.14
    - anthropic>=0.21
    - google-generativeai>=0.4

    # Utilities
    - wandb>=0.16  # Experiment tracking
    - pydantic>=2.6
    - rich>=13  # CLI formatting
```

### 5.3 Training Configuration

#### 5.3.1 SFT Cold Start Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SFTConfig:
    """Configuration for Cold Start SFT phase."""

    # Model
    base_model: str = "meta-llama/Llama-3.3-1B-Instruct"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    num_epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data
    max_seq_length: int = 2048
    packing: bool = True  # Pack multiple examples per sequence

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

    # DeepSpeed
    deepspeed_config: Optional[str] = "configs/ds_config_zero2.json"
```

#### 5.3.2 Dr.GRPO Configuration

```python
@dataclass
class DrGRPOConfig:
    """Configuration for Dr.GRPO alignment phase."""

    # Model
    student_checkpoint: str = "checkpoints/sft_final"
    reference_model: str = "Qwen/Qwen2.5-7B-Instruct"
    freeze_reference: bool = True

    # GRPO Hyperparameters
    group_size: int = 8  # Trajectories per prompt
    kl_coef: float = 0.05
    kl_target: float = 0.1  # Target KL for adaptive coefficient
    clip_range: float = 0.2  # PPO-style clipping

    # Sampling
    sampling_temperature: float = 0.8
    top_p: float = 0.95
    max_response_tokens: int = 512

    # Training
    num_train_steps: int = 5000
    batch_size: int = 4  # Prompts per batch (x group_size trajectories)
    learning_rate: float = 1e-6
    lr_scheduler: str = "constant_with_warmup"
    warmup_steps: int = 100

    # Curriculum
    use_curriculum: bool = True
    curriculum_stages: int = 3
    stage_advancement_threshold: float = 0.8  # Advance when 80% pass

    # Regularization
    entropy_bonus: float = 0.01
    advantage_normalization: bool = True

    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500

    # Wandb
    project_name: str = "mechanex-drgrpo"
    run_name: Optional[str] = None
```

### 5.4 Distributed Training Setup

#### 5.4.1 DeepSpeed Configuration (ZeRO-2)

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

#### 5.4.2 Multi-GPU Launch Script

```bash
#!/bin/bash
# train_drgrpo.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=29500

deepspeed --num_gpus=8 \
    train_drgrpo.py \
    --config configs/drgrpo_config.yaml \
    --deepspeed configs/ds_config_zero2.json \
    --output_dir checkpoints/drgrpo_run_001 \
    --wandb_project mechanex-drgrpo \
    --wandb_run_name "drgrpo_8gpu_kl0.05"
```

---

## 6. Evaluation Framework

### 6.1 Evaluation Dimensions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │   ACCURACY    │  │   LATENCY     │  │    COST       │           │
│  │               │  │               │  │               │           │
│  │ • Format Pass │  │ • TTFT        │  │ • $/1M calls  │           │
│  │ • Grounding   │  │ • Tokens/sec  │  │ • VRAM usage  │           │
│  │ • Type Valid  │  │ • E2E p99     │  │ • Power draw  │           │
│  │ • Semantic    │  │               │  │               │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │  ROBUSTNESS   │  │  REASONING    │  │   SAFETY      │           │
│  │               │  │               │  │               │           │
│  │ • Edge cases  │  │ • CoT quality │  │ • Injection   │           │
│  │ • Adversarial │  │ • Error recov │  │   resistance  │           │
│  │ • OOD inputs  │  │ • Uncertainty │  │ • Refusal on  │           │
│  │               │  │               │  │   OOD tools   │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Benchmark Suite

#### 6.2.1 Core Metrics

```python
@dataclass
class EvaluationMetrics:
    """Core evaluation metrics for student model."""

    # Accuracy Metrics
    format_pass_rate: float      # % responses with valid <think>/<tool_call> structure
    grounding_accuracy: float    # % arguments correctly grounded in context
    type_validity_rate: float    # % responses passing JSON schema validation
    semantic_accuracy: float     # % responses matching Teacher's expected output

    # Latency Metrics (milliseconds)
    time_to_first_token: float   # TTFT
    total_generation_time: float # Full response time
    tokens_per_second: float     # Generation throughput

    # Comparative Metrics
    teacher_parity_score: float  # Semantic similarity to Teacher output (0-1)
    improvement_over_base: float # Delta vs un-finetuned base model

    # Robustness Metrics
    edge_case_accuracy: float    # Accuracy on deliberately difficult inputs
    adversarial_resistance: float # Resistance to prompt injection attempts
    ood_refusal_rate: float      # Rate of appropriate refusal on OOD tools
```

#### 6.2.2 Benchmark Datasets

| Benchmark | Size | Purpose | Source |
|-----------|------|---------|--------|
| **MX-ToolBench-Core** | 2,000 | Standard tool-calling accuracy | Synthetic (Teacher-generated) |
| **MX-EdgeCases** | 500 | Boundary conditions, malformed inputs | Curated adversarial set |
| **MX-Grounding** | 1,000 | Entity extraction & grounding | Enterprise document corpus |
| **MX-MultiTool** | 500 | Multi-tool selection & routing | Complex workflow scenarios |
| **Berkeley Function Calling Leaderboard** | External | Industry comparison | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |

#### 6.2.3 Evaluation Script

```python
class MechanexEvaluator:
    """
    Comprehensive evaluation suite for Mechanex student models.
    """

    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_client: TeacherClient,
        ara_module: ARAModule,
        config: EvalConfig
    ):
        self.student = student_model
        self.teacher = teacher_client
        self.ara = ara_module
        self.config = config

    def evaluate(self, benchmark: Dataset) -> EvaluationReport:
        """
        Run full evaluation suite on benchmark dataset.
        """
        results = []

        for example in tqdm(benchmark, desc="Evaluating"):
            # Generate student response
            start_time = time.perf_counter()
            response = self._generate(example['prompt'])
            generation_time = time.perf_counter() - start_time

            # Compute metrics
            result = EvaluationResult(
                example_id=example['id'],

                # Accuracy
                format_valid=self._check_format(response),
                grounding_score=self._check_grounding(response, example),
                type_valid=self._check_types(response, example['tool_schema']),
                semantic_match=self._check_semantic(response, example),

                # Latency
                generation_time_ms=generation_time * 1000,
                token_count=len(self.tokenizer.encode(response)),

                # ARA Score
                ara_result=self.ara.validate(response, example['prompt'])
            )

            results.append(result)

        return self._aggregate_results(results)

    def compare_to_teacher(self, benchmark: Dataset) -> ComparisonReport:
        """
        Head-to-head comparison with Teacher model on same inputs.
        """
        comparisons = []

        for example in tqdm(benchmark, desc="Comparing"):
            # Generate from both models
            student_response = self._generate(example['prompt'])
            teacher_response = self.teacher.generate(example['prompt'])

            # Compare outputs
            comparison = ResponseComparison(
                student_response=student_response,
                teacher_response=teacher_response,
                student_ara_score=self.ara.validate(student_response, example['prompt']).score,
                teacher_ara_score=self.ara.validate(teacher_response, example['prompt']).score,
                semantic_similarity=self._compute_similarity(student_response, teacher_response),
                student_faster_by=self._compare_latency(student_response, teacher_response)
            )

            comparisons.append(comparison)

        return self._aggregate_comparisons(comparisons)

    def evaluate_robustness(self, adversarial_set: Dataset) -> RobustnessReport:
        """
        Evaluate model robustness on adversarial inputs.
        """
        categories = {
            'prompt_injection': [],
            'schema_mismatch': [],
            'entity_hallucination': [],
            'ood_tool_request': []
        }

        for example in adversarial_set:
            category = example['adversarial_type']
            response = self._generate(example['prompt'])

            passed = self._check_adversarial_resistance(
                response,
                example['expected_behavior'],
                category
            )

            categories[category].append(passed)

        return RobustnessReport(
            prompt_injection_resistance=np.mean(categories['prompt_injection']),
            schema_mismatch_handling=np.mean(categories['schema_mismatch']),
            hallucination_resistance=np.mean(categories['entity_hallucination']),
            ood_refusal_rate=np.mean(categories['ood_tool_request'])
        )
```

### 6.3 Success Criteria

#### 6.3.1 Minimum Viable Performance (MVP)

| Metric | Target | Blocking? |
|--------|--------|-----------|
| Format Pass Rate | ≥ 95% | Yes |
| Grounding Accuracy | ≥ 90% | Yes |
| Type Validity | ≥ 95% | Yes |
| Semantic Accuracy | ≥ 85% | Yes |
| Teacher Parity | ≥ 0.90 | No |
| Latency (p50) | ≤ 200ms | No |
| Latency (p99) | ≤ 500ms | Yes |

#### 6.3.2 Stretch Goals

| Metric | Target | Value |
|--------|--------|-------|
| Semantic Accuracy | ≥ 95% | Matches frontier capability |
| Teacher Parity | ≥ 0.97 | Near-indistinguishable from Teacher |
| Edge Case Accuracy | ≥ 80% | Robust generalization |
| BFCL Ranking | Top 10 | Industry recognition |

---

## 7. Implementation Roadmap

### 7.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION ROADMAP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 0          PHASE 1           PHASE 2          PHASE 3                │
│  Foundation       ARA Module        Training         Production              │
│                                     Pipeline         & Scale                 │
│  ┌─────────┐     ┌─────────┐       ┌─────────┐      ┌─────────┐            │
│  │         │     │         │       │         │      │         │            │
│  │ Infra   │────▶│  ARA    │──────▶│  SFT +  │─────▶│ Deploy  │            │
│  │ Setup   │     │  Dev    │       │ Dr.GRPO │      │ & App   │            │
│  │         │     │         │       │         │      │         │            │
│  └─────────┘     └─────────┘       └─────────┘      └─────────┘            │
│                                                                              │
│  Deliverables:    Deliverables:     Deliverables:    Deliverables:          │
│  • Dev env        • ARA module      • SFT pipeline   • REST API             │
│  • API clients    • Reward sandbox  • Dr.GRPO impl   • Electron app         │
│  • Data schemas   • Teacher prompts • Eval suite     • Mechanex-Mini        │
│                                     • Trained model  • Documentation        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Phase 0: Foundation & Infrastructure

#### Objectives
- Establish development environment and tooling
- Set up API integrations with Teacher models
- Define data schemas and pipeline interfaces

#### Deliverables

| # | Task | Description | Dependencies |
|---|------|-------------|--------------|
| 0.1 | Environment Setup | Conda environment, GPU drivers, Docker | None |
| 0.2 | API Client Library | Unified interface for OpenAI/Anthropic/Google | 0.1 |
| 0.3 | Data Schema Definition | Pydantic models for all data types | 0.1 |
| 0.4 | Experiment Tracking | Weights & Biases integration | 0.1 |
| 0.5 | CI/CD Pipeline | GitHub Actions for testing, linting | 0.1 |
| 0.6 | Documentation Scaffold | Architecture docs, API specs | 0.1-0.5 |

#### Acceptance Criteria
- [ ] All team members can run `make setup` and have working environment
- [ ] API clients successfully authenticate and generate responses
- [ ] CI pipeline passes on all commits to main

---

### 7.3 Phase 1: ARA Module Integration

#### Objectives
- Implement the Automated Reward Architect
- Build secure code execution sandbox
- Create reward function template library

#### Deliverables

| # | Task | Description | Dependencies |
|---|------|-------------|--------------|
| 1.1 | Schema Parser | JSON/NL tool schema → canonical form | 0.3 |
| 1.2 | Teacher Compiler Prompts | Prompt templates for reward generation | 0.2 |
| 1.3 | Code Generation Pipeline | Teacher → Python reward class | 1.1, 1.2 |
| 1.4 | Execution Sandbox | Secure environment for reward execution | 0.1 |
| 1.5 | Reward Template Library | Pre-built validators (format, type, grounding) | 1.3 |
| 1.6 | ARA CLI Tool | Command-line interface for reward compilation | 1.3, 1.4, 1.5 |
| 1.7 | Unit Tests | 90%+ coverage on ARA module | 1.1-1.6 |
| 1.8 | Integration Tests | End-to-end ARA workflow tests | 1.7 |

#### Technical Specifications

**1.1 Schema Parser**
```python
class SchemaParser:
    """
    Parses tool definitions from multiple formats into canonical form.

    Supported inputs:
    - OpenAPI 3.0 spec
    - JSON Schema
    - Natural language description
    - Function signature (Python/TypeScript)
    """

    def parse(self, input_schema: Union[str, dict], format: SchemaFormat) -> CanonicalSchema:
        if format == SchemaFormat.OPENAPI:
            return self._parse_openapi(input_schema)
        elif format == SchemaFormat.JSON_SCHEMA:
            return self._parse_json_schema(input_schema)
        elif format == SchemaFormat.NATURAL_LANGUAGE:
            return self._parse_natural_language(input_schema)
        elif format == SchemaFormat.FUNCTION_SIG:
            return self._parse_function_signature(input_schema)
```

**1.4 Execution Sandbox**
```python
class RewardSandbox:
    """
    Secure sandbox for executing generated reward code.

    Security measures:
    - AST validation (no imports except allowlist)
    - Resource limits (CPU, memory, time)
    - Network isolation
    - Filesystem restrictions
    """

    ALLOWED_MODULES = {'re', 'json', 'typing', 'dataclasses'}
    MAX_EXECUTION_TIME = 5.0  # seconds
    MAX_MEMORY_MB = 256

    def execute_reward(
        self,
        reward_code: str,
        response: str,
        prompt: str
    ) -> RewardResult:
        # Validate AST
        self._validate_code_safety(reward_code)

        # Execute in restricted environment
        with ResourceLimiter(time=self.MAX_EXECUTION_TIME, memory=self.MAX_MEMORY_MB):
            result = self._run_isolated(reward_code, response, prompt)

        return result
```

#### Acceptance Criteria
- [ ] ARA successfully compiles reward functions for 10 diverse tool schemas
- [ ] Generated rewards correctly identify valid/invalid tool calls (95%+ accuracy on test set)
- [ ] Sandbox blocks all attempted security exploits in adversarial test suite
- [ ] CLI tool documented and usable by team members

---

### 7.4 Phase 2: Training Pipeline

#### Objectives
- Implement data generation and filtering pipeline
- Build SFT training infrastructure
- Implement Dr.GRPO algorithm
- Establish evaluation framework

#### Deliverables

| # | Task | Description | Dependencies |
|---|------|-------------|--------------|
| 2.1 | Seed Generator | Teacher-driven prompt generation | 1.6 |
| 2.2 | Trajectory Expander | Assistant 8x expansion with diversity | 2.1 |
| 2.3 | Quality Filter Pipeline | ARA + Teacher validation + dedup | 2.2, 1.6 |
| 2.4 | SFT Trainer | Supervised fine-tuning implementation | 2.3 |
| 2.5 | Dr.GRPO Core | Group-relative policy optimization | 2.4, 1.6 |
| 2.6 | KL Reference Integration | Assistant model as KL anchor | 2.5 |
| 2.7 | Curriculum Scheduler | Progressive difficulty training | 2.5 |
| 2.8 | Evaluation Suite | Benchmark runner + metrics | 2.4 |
| 2.9 | Hyperparameter Sweep | Optimal config identification | 2.5, 2.8 |
| 2.10 | POC Training Run | End-to-end on simulated CRM use case | 2.1-2.8 |

#### Technical Specifications

**2.5 Dr.GRPO Core Implementation**

```python
class DrGRPOTrainer:
    """
    Deep Reasoning Group Relative Policy Optimization.

    Key features:
    - Critic-free (rewards computed within groups)
    - KL-constrained (prevents policy collapse)
    - Curriculum-aware (progressive difficulty)
    """

    def __init__(self, config: DrGRPOConfig):
        self.student = self._load_student(config.student_checkpoint)
        self.reference = self._load_reference(config.reference_model)
        self.reward_fn = self._load_reward(config.reward_function_id)
        self.config = config

        # Freeze reference
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

    def train(self, dataset: Dataset):
        optimizer = AdamW(self.student.parameters(), lr=self.config.learning_rate)
        scheduler = get_scheduler(self.config.lr_scheduler, optimizer, ...)

        for step in range(self.config.num_train_steps):
            batch = self._sample_batch(dataset)
            metrics = self.training_step(batch, optimizer)

            # Logging
            if step % self.config.log_every_n_steps == 0:
                wandb.log(metrics, step=step)

            # Evaluation
            if step % self.config.eval_every_n_steps == 0:
                eval_metrics = self.evaluate()
                wandb.log(eval_metrics, step=step)

                # Curriculum advancement
                if self.config.use_curriculum:
                    if eval_metrics['pass_rate'] > self.config.stage_advancement_threshold:
                        self.curriculum.advance_stage()

            # Checkpointing
            if step % self.config.save_every_n_steps == 0:
                self._save_checkpoint(step)

            scheduler.step()

    def training_step(self, batch: Dict, optimizer: Optimizer) -> Dict[str, float]:
        self.student.train()
        optimizer.zero_grad()

        total_loss = 0.0
        all_metrics = defaultdict(list)

        for prompt, tool_schema in zip(batch['prompts'], batch['tool_schemas']):
            # Sample group of trajectories
            trajectories = self._sample_group(prompt, self.config.group_size)

            # Compute rewards
            rewards = torch.tensor([
                self.reward_fn(t, prompt).score for t in trajectories
            ])

            # Group-relative advantages
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Log probabilities
            student_logprobs = self._compute_logprobs(self.student, trajectories, prompt)
            ref_logprobs = self._compute_logprobs(self.reference, trajectories, prompt)

            # Policy gradient loss
            pg_loss = -(advantages * student_logprobs).mean()

            # KL penalty
            kl = (student_logprobs - ref_logprobs).mean()
            kl_loss = self.config.kl_coef * kl

            # Entropy bonus (encourages exploration)
            entropy = -student_logprobs.mean()
            entropy_bonus = self.config.entropy_bonus * entropy

            # Combined loss
            loss = pg_loss + kl_loss - entropy_bonus
            total_loss += loss

            # Track metrics
            all_metrics['reward'].append(rewards.mean().item())
            all_metrics['kl'].append(kl.item())
            all_metrics['entropy'].append(entropy.item())
            all_metrics['advantage_std'].append(advantages.std().item())

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.max_grad_norm)
        optimizer.step()

        return {k: np.mean(v) for k, v in all_metrics.items()}
```

**2.10 POC Training Run - Simulated CRM Use Case**

```python
# poc_crm_training.py
"""
Proof-of-concept training run on simulated CRM tool-calling task.

Tools:
- update_lead_status(lead_id, status)
- search_leads(query, filters)
- create_lead(name, email, company, source)
- assign_lead(lead_id, sales_rep_id)
"""

CRM_TOOLS = [
    {
        "name": "update_lead_status",
        "description": "Update the status of a lead in the CRM",
        "parameters": {
            "lead_id": {"type": "string", "description": "Unique lead identifier"},
            "status": {
                "type": "string",
                "enum": ["New", "Contacted", "Qualified", "Proposal", "Closed Won", "Closed Lost"]
            }
        },
        "required": ["lead_id", "status"]
    },
    # ... other tools
]

def run_poc():
    # Phase 1: Generate seeds
    seeds = generate_seeds(CRM_TOOLS, num_prompts=1000)

    # Phase 2: Expand trajectories
    trajectories = expand_trajectories(seeds, expansion_factor=8)

    # Phase 3: Filter
    filtered = filter_trajectories(trajectories, ara_threshold=0.85)

    # Phase 4: SFT
    sft_model = train_sft(filtered, epochs=2)

    # Phase 5: Dr.GRPO
    final_model = train_drgrpo(sft_model, filtered, steps=2000)

    # Phase 6: Evaluate
    metrics = evaluate(final_model, CRM_TEST_SET)

    print(f"POC Results:")
    print(f"  Format Pass Rate: {metrics.format_pass_rate:.2%}")
    print(f"  Grounding Accuracy: {metrics.grounding_accuracy:.2%}")
    print(f"  Type Validity: {metrics.type_validity_rate:.2%}")
    print(f"  Semantic Accuracy: {metrics.semantic_accuracy:.2%}")
```

#### Acceptance Criteria
- [ ] POC achieves ≥85% accuracy on simulated CRM benchmark
- [ ] SFT training completes without OOM on 4x A100 setup
- [ ] Dr.GRPO shows consistent reward improvement over SFT baseline
- [ ] Evaluation metrics match success criteria from Section 6.3

---

### 7.5 Phase 3: Production & Scale

#### Objectives
- Build production API and deployment infrastructure
- Develop Electron-based developer application
- Design and train Mechanex-Mini architecture
- Create comprehensive documentation

#### Deliverables

| # | Task | Description | Dependencies |
|---|------|-------------|--------------|
| 3.1 | REST API | FastAPI service for model inference | Phase 2 |
| 3.2 | Model Registry | Version control for trained models | 3.1 |
| 3.3 | Quantization Pipeline | INT8/INT4 model optimization | 3.1 |
| 3.4 | Electron App Shell | Desktop application framework | None |
| 3.5 | Training Wizard UI | Step-by-step model training interface | 3.4, Phase 2 |
| 3.6 | Web Agent Integration | Browser automation for app exploration | 3.4 |
| 3.7 | Mechanex-Mini Architecture | Custom 250-500M tool-calling model | Phase 2 research |
| 3.8 | Mechanex-Mini Pre-training | Foundation training on tool-calling corpus | 3.7 |
| 3.9 | Documentation Portal | User guides, API docs, tutorials | 3.1-3.8 |
| 3.10 | Enterprise Pilot | Deployment with design partner | 3.1-3.9 |

#### Technical Specifications

**3.1 REST API**

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Mechanex Inference API", version="1.0.0")

class InferenceRequest(BaseModel):
    prompt: str
    model_id: str
    tool_schema: Optional[dict] = None
    max_tokens: int = 512
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    response: str
    tool_call: Optional[dict]
    reasoning: Optional[str]
    latency_ms: float
    tokens_generated: int

@app.post("/v1/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Generate tool-calling response from trained Mechanex model.
    """
    model = model_registry.get(request.model_id)
    if not model:
        raise HTTPException(404, f"Model {request.model_id} not found")

    start = time.perf_counter()
    response = model.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    latency = (time.perf_counter() - start) * 1000

    # Parse response
    parsed = parse_tool_response(response)

    return InferenceResponse(
        response=response,
        tool_call=parsed.tool_call,
        reasoning=parsed.reasoning,
        latency_ms=latency,
        tokens_generated=len(tokenizer.encode(response))
    )

@app.post("/v1/train", response_model=TrainingJobResponse)
async def start_training(request: TrainingRequest):
    """
    Start a new model training job.
    """
    job_id = training_service.submit_job(
        tool_schemas=request.tool_schemas,
        context_docs=request.context_docs,
        config=request.training_config
    )

    return TrainingJobResponse(job_id=job_id, status="queued")
```

**3.6 Web Agent Integration**

```python
# app/web_agent.py
"""
Browser automation agent for exploring web applications
and generating tool schemas + training data.
"""

from playwright.async_api import async_playwright
from typing import List, Dict

class WebExplorationAgent:
    """
    Explores web applications to discover API patterns and generate
    tool schemas automatically.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.discovered_actions = []

    async def explore(self, url: str, exploration_depth: int = 3) -> ExplorationResult:
        """
        Autonomously explore a web application.

        Returns:
        - Discovered tool schemas
        - Sample interactions for training data
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            await page.goto(url)

            for _ in range(exploration_depth):
                # Take screenshot and get DOM
                screenshot = await page.screenshot()
                dom = await page.content()

                # Ask LLM to identify interactive elements
                actions = await self._identify_actions(screenshot, dom)

                for action in actions:
                    # Execute action and record
                    result = await self._execute_action(page, action)
                    self.discovered_actions.append({
                        'action': action,
                        'before_state': dom,
                        'after_state': await page.content(),
                        'network_requests': result.network_requests
                    })

            await browser.close()

        # Generate tool schemas from discovered patterns
        tool_schemas = await self._generate_schemas(self.discovered_actions)

        # Generate training examples
        training_data = await self._generate_training_data(
            tool_schemas,
            self.discovered_actions
        )

        return ExplorationResult(
            tool_schemas=tool_schemas,
            training_data=training_data,
            raw_interactions=self.discovered_actions
        )
```

**3.7 Mechanex-Mini Architecture**

```python
# models/mechanex_mini.py
"""
Mechanex-Mini: Custom architecture optimized for tool-calling.

Design principles:
1. Efficient attention for structured output
2. Schema-aware embeddings
3. Constrained decoding support
4. Small but reasoning-capable (250-500M params)
"""

from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn

class MechanexMiniConfig(PretrainedConfig):
    model_type = "mechanex_mini"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,  # GQA for efficiency
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,

        # Tool-calling specific
        schema_embedding_dim: int = 256,
        use_schema_conditioning: bool = True,
        constrained_decoding: bool = True,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # ... etc

class MechanexMiniForToolCalling(PreTrainedModel):
    """
    Mechanex-Mini model with tool-calling head.

    Architecture innovations:
    1. Schema Conditioning: Tool schema embedded and cross-attended
    2. Structured Output Head: Separate heads for reasoning vs tool_call
    3. Grammar-Constrained Generation: Built-in JSON grammar support
    """

    def __init__(self, config: MechanexMiniConfig):
        super().__init__(config)

        # Core transformer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MechanexDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)

        # Schema conditioning
        if config.use_schema_conditioning:
            self.schema_encoder = SchemaEncoder(config)
            self.schema_cross_attention = CrossAttention(config)

        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        tool_schema=None,  # Optional schema conditioning
        **kwargs
    ):
        hidden_states = self.embed_tokens(input_ids)

        # Encode schema if provided
        if tool_schema is not None and self.config.use_schema_conditioning:
            schema_embeds = self.schema_encoder(tool_schema)
        else:
            schema_embeds = None

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                schema_embeds=schema_embeds
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
```

#### Acceptance Criteria
- [ ] REST API handles 100 req/s on single A10G
- [ ] Electron app successfully trains custom model from web exploration
- [ ] Mechanex-Mini achieves 90%+ accuracy with 2x lower latency than Llama-3.3-1B
- [ ] Documentation portal live with complete API reference and tutorials

---

## 8. Risk Assessment & Mitigations

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Dr.GRPO instability** | Medium | High | Implement reward shaping, conservative KL, early stopping |
| **Teacher API rate limits** | High | Medium | Multi-provider fallback, caching, batch optimization |
| **Reward hacking** | Medium | High | Diverse reward components, teacher verification, adversarial testing |
| **Catastrophic forgetting** | Medium | Medium | KL constraint, replay buffer, checkpoint ensembling |
| **VRAM OOM during training** | Medium | Medium | Gradient checkpointing, ZeRO-3, batch size tuning |

### 8.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Low trajectory diversity** | Medium | High | Structured diversity prompts, deduplication, augmentation |
| **Teacher hallucinations in seeds** | Low | High | Multi-round verification, human spot-check |
| **Schema coverage gaps** | Medium | Medium | Systematic schema enumeration, edge case generation |

### 8.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Cloud GPU availability** | Medium | High | Multi-cloud strategy, reserved instances, spot fallback |
| **Model serving latency** | Low | Medium | Quantization, batching, hardware selection |
| **Security vulnerabilities** | Low | Critical | Sandbox validation, input sanitization, audit logging |

### 8.4 Risk Response Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                    RISK RESPONSE MATRIX                          │
├───────────────┬──────────────────────────────────────────────────┤
│               │              IMPACT                              │
│               ├──────────┬───────────┬───────────┬──────────────┤
│               │   Low    │  Medium   │   High    │   Critical   │
├───────────────┼──────────┼───────────┼───────────┼──────────────┤
│ P  │ High     │ Monitor  │ Mitigate  │ Mitigate  │ Avoid/Trans  │
│ R  ├──────────┼──────────┼───────────┼───────────┼──────────────┤
│ O  │ Medium   │ Accept   │ Monitor   │ Mitigate  │ Mitigate     │
│ B  ├──────────┼──────────┼───────────┼───────────┼──────────────┤
│    │ Low      │ Accept   │ Accept    │ Monitor   │ Mitigate     │
└────┴──────────┴──────────┴───────────┴───────────┴──────────────┘
```

---

## 9. Technical References

### 9.1 Core Papers

| Paper | Relevance | Link |
|-------|-----------|------|
| **DeepSeek-R1** (2025) | GRPO algorithm foundation | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| **GRPO: Group Relative Policy Optimization** | Original GRPO formulation | [DeepSeek Technical Report](https://github.com/deepseek-ai/DeepSeek-R1) |
| **Constitutional AI** (Anthropic) | Reward modeling principles | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| **Toolformer** (Meta) | Tool-calling in LLMs | [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) |
| **LoRA** | Parameter-efficient fine-tuning | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **DPO** | Direct preference optimization | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |

### 9.2 Relevant Benchmarks

| Benchmark | Description | Link |
|-----------|-------------|------|
| **Berkeley Function Calling Leaderboard** | Industry-standard tool-calling eval | [gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| **API-Bank** | API usage benchmark | [arXiv:2304.08244](https://arxiv.org/abs/2304.08244) |
| **ToolBench** | Large-scale tool benchmark | [arXiv:2305.16504](https://arxiv.org/abs/2305.16504) |

### 9.3 Implementation Resources

| Resource | Purpose | Link |
|----------|---------|------|
| **TRL (Transformers RL)** | RL training library | [github.com/huggingface/trl](https://github.com/huggingface/trl) |
| **DeepSpeed** | Distributed training | [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| **vLLM** | Efficient inference | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **Outlines** | Structured generation | [github.com/outlines-dev/outlines](https://github.com/outlines-dev/outlines) |

---

## 10. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ARA** | Automated Reward Architect - module that generates reward functions from tool specs |
| **Dr.GRPO** | Deep Reasoning Group Relative Policy Optimization - critic-free RL algorithm |
| **Golden Trace** | High-quality reasoning + tool-call example generated by Teacher |
| **Grounding** | Ensuring generated arguments are derived from input context |
| **KL Divergence** | Measure of difference between student and reference policy |
| **SFT** | Supervised Fine-Tuning - initial training phase on curated examples |
| **Trajectory** | Complete model response including reasoning and tool call |

### Appendix B: Configuration Templates

#### B.1 Tool Schema Template
```json
{
  "name": "tool_name",
  "description": "Human-readable description of what this tool does",
  "parameters": {
    "param1": {
      "type": "string",
      "description": "Description of parameter",
      "required": true
    },
    "param2": {
      "type": "integer",
      "description": "Description of parameter",
      "minimum": 0,
      "maximum": 100
    }
  },
  "returns": {
    "type": "object",
    "properties": {
      "result": {"type": "string"}
    }
  }
}
```

#### B.2 Training Configuration Template
```yaml
# config/training_template.yaml
experiment:
  name: "experiment_name"
  project: "mechanex"

data:
  seed_prompts: 5000
  expansion_factor: 8
  filter_threshold: 0.85

sft:
  base_model: "meta-llama/Llama-3.3-1B-Instruct"
  epochs: 2
  batch_size: 8
  learning_rate: 2e-5

drgrpo:
  group_size: 8
  kl_coef: 0.05
  num_steps: 5000
  learning_rate: 1e-6

evaluation:
  benchmarks:
    - "mx_core"
    - "mx_edge_cases"
  metrics:
    - "format_pass_rate"
    - "grounding_accuracy"
    - "semantic_accuracy"
```

### Appendix C: API Reference Stubs

See separate `API_REFERENCE.md` document (to be generated in Phase 3).

### Appendix D: Team Responsibilities

| Role | Primary Responsibilities | Phase Focus |
|------|-------------------------|-------------|
| **ML Engineer (Senior)** | Dr.GRPO implementation, training pipeline | Phase 2 |
| **ML Engineer** | Data pipeline, ARA module | Phase 1-2 |
| **Backend Engineer** | API, infrastructure, deployment | Phase 0, 3 |
| **Frontend Engineer** | Electron app, training wizard | Phase 3 |
| **Research Engineer** | Mechanex-Mini architecture, evaluation | Phase 2-3 |
| **DevOps** | CI/CD, cloud infrastructure, monitoring | All phases |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-10 | Axionic Labs Engineering | Initial release |

---

*This document is confidential and intended for Axionic Labs internal use only.*
