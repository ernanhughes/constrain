# Constrain 🛡️

**Policy-bounded AI for high-trust environments**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/research-hallucination--energy-green)](https://github.com/ernanhughes/certum)

> **Generation may be stochastic. Acceptance must be deterministic.**

Constrain is a research-to-production framework for bounding stochastic LLM generation using **deterministic software enforcement**. Instead of trying to “train truth into the model”, Constrain treats the LLM as an untrusted generator and moves reliability downstream into a **policy + control system** that decides what is allowed to commit.

---

## Why Constrain exists

Hallucinations rarely appear as a single bad token. In practice they behave like **runaway drift**: once a reasoning trajectory leaves a grounded region, it tends to keep drifting unless the system actively **rolls back** or **resets**.

Constrain’s goal is to:

1. **measure** drift continuously (Hallucination Energy),
2. **detect** runaway conditions early (energy + slope),
3. **recover** deterministically (stack-based rollback),
4. **evaluate** policies fairly (global collapse definition).

---

## Core principle

| Traditional LLM reliability | Constrain                       |
| --------------------------- | ------------------------------- |
| Make the model reliable     | Make the **system** disciplined |
| Trust model confidence      | Trust executable policy         |
| Binary “hallucinated / not” | Continuous grounding metric     |
| Model-centric fixes         | Architecture-centric control    |

---

## Architecture (four layers)

```
┌─────────────────────────────────────────────────────────────────┐
│ MEASUREMENT (global, immutable)                                 │
│  • Hallucination Energy = projection residual                    │
│  • Energy slope ΔE                                               │
│  • Violation level via global thresholds                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ POLICY / CONTROL (deterministic)                                │
│  • Input: EnergyDynamics + temperature                           │
│  • Output: ALLOW / ROLLBACK / RESET / TERMINATE                  │
│  • Policies respond to signals; they do not redefine collapse     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STATE (stack-based)                                             │
│  • push(): commit a step                                         │
│  • pop(): remove bad steps (true rollback)                        │
│  • reset(): clear to initial prompt                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ EVALUATION (global)                                             │
│  • Global collapse definition (containment failure)               │
│  • Survival depth, intervention effectiveness                     │
│  • Comparable across policies                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security analogy (how to explain it to anyone)

Constrain treats policy like an **OS security module**:

| OS security concept     | Constrain equivalent           |
| ----------------------- | ------------------------------ |
| Application (untrusted) | LLM reasoning process          |
| Syscall                 | “Generate next step”           |
| Kernel security module  | Policy / control engine        |
| Permission violation    | Energy exceeds threshold       |
| Audit log               | Run + step persistence         |
| System compromise       | Collapse (containment failure) |

**Key principle:** the application cannot redefine what a violation is. The kernel defines violations globally. Policies only decide how to respond.

---

## Hallucination Energy (core metric)

Hallucination Energy measures how far a claim extends beyond the semantic span of its evidence.

Given:

* claim embedding **c** ∈ ℝᵈ
* evidence embeddings **E** = {**e**₁…**e**ₙ}

Let **Uᵣ** be a truncated orthonormal basis for span(**E**) (via SVD):

```text
ĉ = UᵣUᵣᵀc
r = c - ĉ
H(c, E) = ||r||₂ / ||c||₂
```

Interpretation (default bands; calibrate per domain):

| Band           |          Energy | Meaning                         |
| -------------- | --------------: | ------------------------------- |
| stable         |        < τ_soft | grounded / contained            |
| drift          | τ_soft–τ_medium | mild deviation                  |
| unstable       | τ_medium–τ_hard | significant unsupported content |
| hard violation |        > τ_hard | likely runaway                  |

---

## Collapse definition (global)

Constrain uses a **control-style** definition of collapse:

> **Collapse = persistent hard violation + rising trend.**
> i.e. energy is high and stays high, and the slope remains positive often enough that the controller cannot regain containment.

Operationally (defaults):

* `consecutive_hard >= collapse_hard_n` **OR**
* `consecutive_rising >= collapse_rising_n` (while in violation bands)

This definition is **global** and used for **all** policy comparisons.

---

## State machine (why rollback actually works)

Constrain’s state is a **stack**. Rollback removes states; it does not merely “point backward”.

```python
state.push("step 1")
state.push("step 2 (bad)")
state.pop()
assert state.current == "step 1"
```

This is the core fix for runaway hallucination cascades: bad steps don’t remain in memory.

---

## Attempts vs iterations (important)

Constrain tracks two counters:

* **attempt**: increments on every model call (including retries)
* **iteration**: increments only when a step is **committed** (ALLOW/push)

This avoids survival-analysis confusion: retries don’t artificially inflate depth.

---

## Quick start

### Install

```bash
git clone https://github.com/ernanhughes/constrain.git
cd constrain

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -e .
cp constrain.toml.example constrain.toml
```

### Run an experiment

```python
from constrain.runner import run

# Baseline (no control)
run_id = run(policy_id=0, seed=42, num_problems=20, num_recursions=6)

# Control policy (energy + slope)
run_id = run(policy_id=5, seed=42, num_problems=20, num_recursions=6)
```

### CLI (examples)

```bash
# Run experiment (adjust module name if you use a different entrypoint)
py -m constrain.main --policy-id 5 --num-problems 20

# Evaluate recent runs
py -m constrain.services.policy_evaluation_service --recent

# Intervention timing diagnostics
py -m constrain.services.intervention_timing_service --recent
```

> If a CLI module name differs in your codebase, update it here (this README is meant to match the repo exactly).

---

## Cached runs (offline testing)

Constrain can reuse cached reasoning steps to test policy/control loops without calling an LLM (useful for unit tests and replay).

**Recommended workflow**

1. Run once with full model calls to populate cache.
2. Re-run policies over the same problems using cached-only mode.

> If your runner exposes a `cached_only=True` flag, document it here; otherwise remove this section to avoid drift.

---

## Project structure (high-level)

```
constrain/
├── energy/            # energy computation + embedding cache
├── control/           # deterministic control policy (energy+slope)
├── policy/            # policy registry (legacy + migration target)
├── reasoning_state.py # stack-based state machine + persistence
├── runner.py          # experiment orchestration
├── services/          # evaluation + analysis services
├── analysis/          # scientific analysis & aggregation
└── data/              # ORM/DTO/stores for persistence
```

---

## Configuration

`constrain.toml` (example):

```toml
[experiment]
num_problems = 200
num_recursions = 10
initial_temperature = 1.1

[tau]
soft = 0.30
medium = 0.50
hard = 0.70

[policy]
min_temperature = 0.1
revert_cooldown_factor = 0.9
aggressive_cooldown_factor = 0.75
reset_cooldown_factor = 0.7

[control]
collapse_hard_n = 3
collapse_rising_n = 2
runaway_slope_eps = 0.05
```

---

## Policy comparison (what we measure)

All policies share the **same collapse definition**, enabling fair comparison.

| Metric            | Meaning                               | Type            |
| ----------------- | ------------------------------------- | --------------- |
| energy AUC        | energy → future collapse prediction   | global          |
| collapse rate     | fraction of problems that collapse    | global          |
| survival depth    | committed iterations before collapse  | global          |
| intervention rate | fraction of steps with action ≠ ALLOW | policy-specific |
| recovery success  | post-intervention improvement rate    | policy-specific |

---

## Development

```bash
pip install -e ".[dev]"
pytest -v
black constrain/
isort constrain/
```

---

## Contributing

Contributions are welcome, especially:

* additional policies/controllers
* better collapse definitions per task family
* causal evaluation + counterfactual replay tooling
* performance work (batching + embedding caching)

---

## License

Apache 2.0 — see `LICENSE`.

---

## Contact

* **Author:** Ernan Hughes
* **Email:** [ernanhughes@gmail.com](mailto:ernanhughes@gmail.com)
* **GitHub:** [https://github.com/ernanhughes](https://github.com/ernanhughes)

