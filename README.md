# Thai-MAC — Hierarchical Multi-Agent Coordination for Active Multi-Camera Target Coverage

> A hierarchical multi-agent reinforcement-learning agent that coordinates a team
> of pan/zoom cameras to **actively track multiple moving targets** in a cluttered
> 2D world. Built on the [**HiT-MAC**][hitmac] formulation and the
> [**MATE**][mate] (Multi-Agent Tracking Environment) benchmark, and trained with
> asynchronous advantage actor-critic (**A3C**).

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.7%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c.svg">
  <img alt="RL" src="https://img.shields.io/badge/RL-A3C%20%7C%20MARL%20%7C%20Hierarchical-success.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Why this problem (Robotics & RL)](#why-this-problem-robotics--rl)
- [The environment in detail](#the-environment-in-detail)
- [Method: a two-level coordination hierarchy](#method-a-two-level-coordination-hierarchy)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Monitoring & checkpoints](#monitoring--checkpoints)
- [Design highlights](#design-highlights)
- [References](#references)
- [License](#license)

---

## Overview

Coordinating many sensors or robots to **collectively perceive a dynamic scene**
is a recurring problem in robotics: a fleet of agents with *limited, directional*
fields of view must continuously decide **what to look at** and **how to point**
so that, as a team, they keep as many moving targets in view as possible.

This repository tackles that problem as a cooperative **multi-agent reinforcement
learning (MARL)** task and solves it with a **hierarchical** policy that separates
*high-level assignment* from *low-level control*:

| Level | Policy | Decides | Action type | Timescale |
|-------|--------|---------|-------------|-----------|
| **Coordinator** | `A3C_Multi` | *Which* targets each camera should attend to | Discrete goal map | Coarse (every macro-step) |
| **Executor** | `A3C_Single` | *How* each camera should pan / zoom | Continuous control | Fine (inner control loop) |

The decomposition mirrors the **HiT-MAC** ("Hierarchical Target-oriented
Multi-Agent Coordination") framework of Xu, Zhong & Wang (NeurIPS 2020), and runs
on the **MATE** active-tracking benchmark (Pan et al., NeurIPS 2022).

---

## Why this problem (Robotics & RL)

This task sits squarely at the intersection of **robotics** and **reinforcement
learning**, which is exactly why it is a useful showcase of both skill sets:

**Robotics / active perception**

- **Directional sensor networks & PTZ control.** Each agent is a camera with a
  bounded, steerable field of view — the same abstraction used for pan-tilt-zoom
  surveillance rigs, multi-robot active SLAM, and coverage control of mobile
  sensor teams.
- **Active sensing under partial observability.** Agents must *act to perceive*:
  pointing decisions change what is observable next step, a hallmark of robotics
  perception-action loops.
- **Decentralized execution.** Each camera acts on its own local observation,
  matching the communication and compute constraints of real multi-robot systems.

**Reinforcement learning**

- **Cooperative MARL** with a shared team objective (coverage rate) and the
  associated **multi-agent credit-assignment** challenge.
- **Hierarchical RL / temporal abstraction.** A high-level coordinator commits to
  a target assignment that a low-level controller executes over several primitive
  steps — an options-style hierarchy.
- **Shapley-value credit assignment.** The coordinator's critic estimates each
  agent's *marginal contribution* to the team value, providing a principled
  decomposition of the shared reward.
- **Attention over variable-sized entity sets**, so a single policy generalizes
  across a changing number of visible targets and obstacles.
- **Distributed, asynchronous training (A3C)** with a shared parameter server.

---

## The environment in detail

The agent is trained and evaluated on the **MATE — Multi-Agent Tracking
Environment** ([XuehaiPan/mate][mate]), wrapped here as two task-specific Gym-style
environments. MATE is an *asymmetric two-team zero-sum stochastic game with partial
observation*: a team of **cameras** tries to maximize coverage of a team of moving
**targets** that try to evade them, amid static **obstacles** that occlude lines of
sight. In this project the camera team is learned while the target team follows a
fixed `GreedyTargetAgent`, turning the problem into a well-defined cooperative
control task for the cameras.

The default configuration used here is **4 cameras vs. 8 targets with 9 obstacles**.

### Entities

| Entity | Count (default) | Role |
|--------|-----------------|------|
| Camera (agent) | 4 | Steerable, limited-FoV sensor — the controllable team |
| Target | 8 | Moving objects to be kept in view (evasive `GreedyTargetAgent`) |
| Obstacle | 9 | Static occluders that block sight lines and constrain motion |

### Observation space

Each camera receives a **local, egocentric observation vector** that packs:

- its **own state** — position `(x, y)`, orientation, current field-of-view angle,
  and sensing parameters;
- a **per-target block** — 5 features per target `[x, y, …, …, visible_flag]`;
- a **per-obstacle block** — 4 features per obstacle `[x, y, …, visible_flag]`.

The parsing logic in [`hitmac/observation.py`](hitmac/observation.py) slices this
flat vector into structured `[num_cameras, num_entities, num_features]` tensors and,
for visible entities, re-expresses positions in the **camera's own frame** — a
small but important inductive bias for egocentric control. The `visible_flag`
encodes **partial observability**: an entity occluded by an obstacle or outside the
field of view is simply marked invisible.

### Action space

- **Coordinator** — a **binary goal map** of shape `[num_cameras, num_targets]`.
  Entry `(i, j) = 1` means "camera *i* is responsible for target *j*". This is a
  per-pair `Discrete(2)` decision produced by the coordinator policy.
- **Executor** — a **continuous 2-D control** per camera: a *rotation* command and
  a *field-of-view (zoom)* command, squashed with `tanh`/`softplus` and clipped to
  the environment's action bounds.

### Reward

- **Team coverage (coordinator).** The coordinator is rewarded by the realized
  `real_coverage_rate` — the fraction of targets actually covered by the team —
  averaged over the inner control horizon.
- **Shaped tracking reward (executor).** Each camera receives a dense,
  per-target signal `1 − |angle_h| / fov` that rewards keeping an *assigned and
  visible* target close to the optical axis, and penalizes losing it. This shaping
  makes the low-level continuous-control problem learnable.

### Temporal structure

The coordinator operates at a **coarser timescale** than the executor: a single
coordinator decision (goal map) is held fixed while the low-level controllers are
rolled forward for several primitive steps (`keep = 10`). This temporal abstraction
is what makes the hierarchy effective — the coordinator reasons about *assignment*,
not *micro-control*.

---

## Method: a two-level coordination hierarchy

```
                    ┌───────────────────────────────────────────────┐
                    │  Joint camera observations  (partial, local)   │
                    └───────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              ▼                                                      ▼
  ┌───────────────────────────┐                      ┌──────────────────────────────┐
  │   COORDINATOR (A3C_Multi)  │                      │     EXECUTOR (A3C_Single)     │
  │                            │                      │                               │
  │  MLP encoder over          │   goal map           │  Attention encoders over      │
  │  camera×target pairs       │  [cams × targets]    │  targets & obstacles          │
  │  → self-attention          │ ───────────────────► │  → Gaussian policy            │
  │  → discrete goal policy    │   (who tracks whom)  │  → (rotation, zoom) control    │
  │  → Shapley-value critic    │                      │  → state-value critic          │
  └───────────────────────────┘                      └──────────────────────────────┘
              │                                                      │
              │  team coverage reward          per-camera shaped tracking reward
              └──────────────────────────────────────────────────────┘
                                        │
                              A3C + GAE update (shared optimizer, N async workers)
```

**Coordinator — `A3C_Multi`** ([`hitmac/model.py`](hitmac/model.py))
Encodes every camera–target pair, applies **self-attention** to reason about the
joint assignment, and outputs a per-pair on/off decision. Its critic uses
`AMCValueNet`, a **Shapley-value** decomposition that estimates each agent's
marginal contribution to the team value by attending over the coalition of
preceding agents — a principled answer to multi-agent credit assignment.

**Executor — `A3C_Single`** ([`hitmac/model.py`](hitmac/model.py))
Encodes the targets and obstacles seen by a single camera with **permutation-
invariant attention** ([`AttentionLayer`](hitmac/modules.py)), then emits a
continuous `(rotation, zoom)` action through a Gaussian policy. The same network
can drive the cameras inside the coordinator environment once trained, replacing
the hand-crafted baseline controller.

**Training — A3C + GAE** ([`hitmac/train.py`](hitmac/train.py), [`hitmac/agent.py`](hitmac/agent.py))
Both levels are trained with asynchronous advantage actor-critic: multiple worker
processes collect rollouts in parallel and push gradients to a shared model via a
shared optimizer ([`hitmac/shared_optim.py`](hitmac/shared_optim.py)). Advantages
use **Generalized Advantage Estimation**, with entropy regularization and gradient
clipping for stability.

---

## Repository structure

```
Thai-MAC/
├── hitmac/                     # main package
│   ├── main.py                 # training entry point (launches workers + evaluator)
│   ├── arguments.py            # command-line interface
│   ├── train.py                # A3C training worker
│   ├── evaluate.py             # evaluation + checkpointing process
│   ├── agent.py                # rollout buffer + A3C/GAE update
│   ├── model.py                # Coordinator (A3C_Multi) & Executor (A3C_Single)
│   ├── modules.py              # attention, noisy linear & recurrent building blocks
│   ├── shared_optim.py         # shared-memory Adam / RMSprop for A3C
│   ├── observation.py          # MATE observation parsing & target localization
│   ├── environment.py          # environment factory
│   ├── env_coordinator.py      # Pose-v1: high-level target-assignment env
│   ├── env_executor.py         # Pose-v0: low-level per-camera control env
│   └── utils.py                # logging, init & gradient-sharing helpers
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Requires **Python 3.7+** and PyTorch.

```bash
# 1. Clone this repository
git clone https://github.com/tranthai189765/Thai-MAC.git
cd Thai-MAC

# 2. Install the core dependencies
pip install -r requirements.txt

# 3. Install the MATE environment (from source)
git config --global core.symlinks true   # Windows only
pip install "git+https://github.com/XuehaiPan/mate.git#egg=mate"
```

> **Note.** MATE is a third-party benchmark and is intentionally **not vendored**
> into this repository (it is git-ignored). Install it separately as shown above.

---

## Usage

All commands are run as a module from the repository root.

### 1. Train the Executor (low-level pan/zoom control)

```bash
python -m hitmac.main --env Pose-v0 --model single-att --workers 6
```

### 2. Train the Coordinator (high-level target assignment)

```bash
python -m hitmac.main --env Pose-v1 --model multi-att-shap --workers 6
```

### 3. Evaluate the full hierarchy with trained checkpoints

```bash
python -m hitmac.main --env Pose-v1 --model multi-att-shap --workers 0 \
    --load-coordinator-dir trainedModel/best_coordinator.pth \
    --load-executor-dir   trainedModel/best_executor.pth
```

Here the coordinator drives the goal assignment while the trained executor
provides the low-level camera control, so the command exercises **both levels of
the hierarchy together**.

### Useful flags

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | `1` | Number of asynchronous A3C workers (`0` = evaluation only) |
| `--lr` | `5e-4` | Learning rate |
| `--gamma` | `0.9` | Discount factor |
| `--tau` | `1.0` | GAE λ |
| `--entropy` | `0.01` | Entropy regularization coefficient |
| `--num-steps` | `20` | Forward steps per A3C update |
| `--gpu-ids` | `-1` | GPU ids (`-1` = CPU) |
| `--norm-reward` | off | Online reward standardization |
| `--log-dir` | `logs/` | TensorBoard logs & checkpoints |

Run `python -m hitmac.main --help` for the full list.

> The model variant is selected from the `--model` string: names containing
> `single` build the **executor** (`A3C_Single`), names containing `multi` build
> the **coordinator** (`A3C_Multi`); add `shap` to use the **Shapley-value**
> critic. Environment ids containing `v0` select the executor environment,
> otherwise the coordinator environment is used.

---

## Monitoring & checkpoints

- **TensorBoard.** Training and evaluation scalars (policy/value loss, entropy,
  per-agent reward, episode length, FPS) are written under `--log-dir`:

  ```bash
  tensorboard --logdir logs/
  ```

- **Checkpoints.** The evaluation process saves the best-so-far model as
  `best.pth` and the latest snapshot as `new.pth` inside the run's log directory.

---

## Design highlights

- **Clean two-level architecture** that cleanly separates assignment from control,
  with each level a self-contained actor-critic module.
- **Attention everywhere** — set-based encoders make the policies invariant to the
  ordering and (within limits) the number of targets/obstacles.
- **Shapley-value critic** for transparent multi-agent credit assignment.
- **Egocentric, partially observable inputs** parsed into structured tensors,
  faithfully reproducing the perception constraints of a real sensor team.
- **Distributed A3C** with shared-memory optimizers for efficient CPU training.

---

## References

- Jing Xu, Fangwei Zhong, Yizhou Wang. *Learning Multi-Agent Coordination for
  Enhancing Target Coverage in Directional Sensor Networks* (HiT-MAC). **NeurIPS
  2020.** [[paper]][hitmac]
- Xuehai Pan, Mickel Liu, Fangwei Zhong, Yaodong Yang, Song-Chun Zhu, Yizhou Wang.
  *MATE: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target
  Coverage Control.* **NeurIPS 2022 (Datasets & Benchmarks).** [[code]][mate]

```bibtex
@inproceedings{xu2020hitmac,
  title     = {Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks},
  author    = {Xu, Jing and Zhong, Fangwei and Wang, Yizhou},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}

@inproceedings{pan2022mate,
  title     = {{MATE}: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control},
  author    = {Pan, Xuehai and Liu, Mickel and Zhong, Fangwei and Yang, Yaodong and Zhu, Song-Chun and Wang, Yizhou},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022}
}
```

---

## License

Released under the [MIT License](LICENSE). The MATE environment is distributed
under its own license by its respective authors.

[hitmac]: https://arxiv.org/pdf/2010.13110
[mate]: https://github.com/XuehaiPan/mate
