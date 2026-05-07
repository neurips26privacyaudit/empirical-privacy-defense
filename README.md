# Black-Box Auditing of DP-SGD

A framework for empirically auditing the privacy guarantees of Differentially Private Stochastic Gradient Descent (DP-SGD) via membership inference attacks (MIA). This codebase supports a range of canary/attack designs and gradient-filtering defenses, and provides tools to compute tight empirical lower bounds on the privacy parameter ε.

## Overview

DP-SGD offers formal privacy guarantees, but those guarantees may not be tight in practice. This framework audits DP-SGD by:

1. Training many shadow model pairs — one with a "canary" sample, one without
2. Running a membership inference attack to score whether the canary was included
3. Computing an empirical lower bound on ε from the attack success rate

Supported canary/attack types, datasets, models, and defenses are listed below.

## Installation

```bash
conda env create -f env.yaml
conda activate bb_audit_dpsgd
```

Requires CUDA 12.1 and PyTorch 2.5.1. See `env.yaml` for the full dependency list.

## Repository Structure

```
bb-audit-dpsgd-clean/
├── models/                         # Model architectures (CNN, WideResNet, MLP, LSTM, LR)
├── utils/
│   ├── audit.py                    # Empirical epsilon computation (GDP / CP methods)
│   ├── canaries.py                 # Canary crafting utilities (FGSM, gradient-space)
│   ├── dpsgd.py                    # DP-SGD training loop, per-sample gradients, DefenseConfig
│   ├── data.py                     # Dataset loading (MNIST, CIFAR-10/100, Purchase, text)
│   ├── training.py                 # Model init, augmentation, evaluation
│   ├── args.py                     # Shared argument parser
│   └── accounting.py              # DP noise multiplier calculation
├── parallel_audit_model.py         # Main distributed audit (torchrun)
├── parallel_audit_multi_canary.py  # Multi-canary audit
├── parallel_audit_model_global_filter.py  # Audit with global gradient filtering
├── parallel_audit_model_hamp.py    # Label-only audit with HAMP defense
├── audit_model.py                  # Single-GPU audit
├── generate_gradient_cancelling_attack.py
├── generate_gradient_bandwidth_attack.py
├── generate_dense_gradient_canary.py
├── generate_input_cancelling_canaries.py
├── defense_aware_canary.py         # Bilevel canary optimization against filtering defenses
├── compute_empirical_epsilon.py    # Compute ε lower bound from audit results
├── compute_emp_eps_cp.py           # Conservative Poisson ε estimation
├── print_tradeoff.py               # Summarize privacy-utility tradeoff curves
├── audit_all_samples.py            # Audit every training sample (not just canaries)
├── fairness_audit.py               # Fairness impact analysis (colored MNIST)
├── scripts/                        # SLURM job scripts and interactive run commands
├── data/                           # Datasets (Purchase, TinyShakespeare)
├── target_samples/                 # Pre-computed canary samples
└── tradeoff_curves/                # Output: aggregated experiment results
```

## Supported Configurations

**Datasets:** `mnist`, `cifar10`, `cifar100`, `purchase`, `tiny_shakespeare`

**Models:** `cnn`, `wideresnet`, `mlp`, `lstm`, `lr`

**Canary / attack types:**
- `blank` — zero or interpolated clean image
- `mislabeled` — real image with an incorrect label
- `fgsm` — fast gradient sign method adversarial example
- `clipbkd` — clipping backdoor (adversarial perturbation scaled to norm budget)
- `badnets` — BadNets backdoor trigger
- `gradient_cancel` — two groups of canaries with canceling gradients
- `gradient_bandwidth` — canaries targeting gradient magnitude limits
- `gradient_space_canary` — dense one-hot gradient canaries

**Defenses:**
- Gradient-norm filtering (`--defense`, `--defense_score_fn`, `--defense_k`)
- Global gradient filtering (`parallel_audit_model_global_filter.py`)
- HAMP — entropy regularization + confidence randomization (`parallel_audit_model_hamp.py`)

## Command Builder

`craft_command.py` is an interactive CLI wizard that builds `parallel_audit_model.py` commands for you. Rather than hand-writing long argument lists, it walks you through every option, fills in dataset-specific defaults, and emits a ready-to-run command or a `.slurm` file.

```bash
python craft_command.py
```

It supports three output modes:

| Mode | Description |
|---|---|
| `1` | Generate a `.slurm` batch file (with `#SBATCH` headers) |
| `2` | Generate a `srun + torchrun` command for a multi-node `idev` session |
| `3` | Generate a standalone `torchrun` command for a single-GPU `idev` session |

The wizard steps through: dataset/model selection, training hyperparameters, DP privacy budget, canary type, audit configuration, defense settings, and output directory. It suggests a sensible output path based on your choices and lets you save the final command to a file.

## Usage

### Distributed Audit (Multi-GPU, recommended)

Uses `torchrun` for distribution across nodes. Typical invocation on a SLURM cluster:

```bash
torchrun --nnodes=5 --nproc_per_node=1 parallel_audit_model.py \
  --data_name mnist --model_name cnn \
  --n_reps 400 --n_epochs 100 \
  --lr 3 --batch_size 4000 --block_size 4000 \
  --epsilon 10.0 --delta 1e-5 --max_grad_norm 1.0 \
  --target_type mislabeled \
  --out exp_data/mnist_mislabeled_eps10/
```

### Single-GPU Audit

```bash
python parallel_audit_model.py \
  --data_name cifar10 --model_name cnn \
  --n_reps 200 --n_epochs 100 \
  --batch_size 512 --block_size 256 \
  --epsilon 8.0 --defense --defense_k 5 \
  --target_type clipbkd \
  --out exp_data/cifar10_clipbkd_defense/
```

Set `--epsilon` to `None` to train without DP.

### Gradient Cancelling Canary Generation

```bash
python generate_gradient_cancelling_attack.py \
  --data_name mnist --model_name cnn \
  --n_epochs 100 --lr 3 --batch_size 4000 \
  --epsilon 10.0 --max_grad_norm 1.0 \
  --n_group_a 50 --n_group_b 50 \
  --alpha 0.1 --beta 0.1 \
  --output target_samples/grad_cancel_canaries.pt
```

### Defense-Aware Canary Optimization

Two-phase bilevel optimization: first collect the training trajectory, then optimize canaries to evade gradient-norm filtering.

```bash
python defense_aware_canary.py \
  --data_name mnist --model_name cnn \
  --n_epochs 100 --lr 3 --batch_size 4000 \
  --epsilon 10.0 --defense_k 5 \
  --bilevel --n_outer 50 --n_inner 3 \
  --lam 0.01 0.1 0.5 1.0 \
  --output target_samples/defense_aware_canaries.pt
```

### Multi-Canary Audit

Tests cumulative privacy loss when multiple canaries are included per shadow model:

```bash
torchrun --nnodes=5 --nproc_per_node=1 parallel_audit_multi_canary.py \
  --data_name mnist --model_name cnn \
  --n_reps 400 --n_canaries 50 \
  --epsilon 10.0 --target_type mislabeled \
  --out exp_data/multi_canary/
```

### Compute Empirical ε

```bash
python compute_empirical_epsilon.py exp_data/mnist_mislabeled_eps10/ \
  --alpha 0.05 --delta 1e-5
```

### Summarize Privacy-Utility Tradeoff

```bash
python print_tradeoff.py exp_data/tradeoff_curves/
```

## Key Arguments

| Argument | Description |
|---|---|
| `--data_name` | Dataset: `mnist`, `cifar10`, `cifar100`, `purchase`, `tiny_shakespeare` |
| `--model_name` | Model: `lr`, `cnn`, `wideresnet`, `mlp`, `lstm` |
| `--target_type` | Canary/attack type (see list above) |
| `--epsilon` | DP privacy budget ε (`None` = non-private) |
| `--delta` | DP delta (default: `1e-5`) |
| `--max_grad_norm` | Per-sample gradient clipping norm |
| `--n_reps` | Number of shadow model pairs (more = tighter bound) |
| `--n_epochs` | Training epochs per shadow model |
| `--defense` | Enable gradient-norm filtering defense |
| `--defense_k` | Number of samples to filter per class per step |
| `--defense_score_fn` | Scoring function: `grad_norm`, `loss_volatility`, `grad_dir_volatility`, etc. |
| `--defense_score_norm` | Norm for scoring: `l2`, `linf`, `l1` |
| `--out` | Output directory for scores, losses, and metadata |

## Output Format

Each audit run saves to the `--out` directory:

| File | Contents |
|---|---|
| `scores_in.npy` | Attack scores for in-world shadow models (shape: `n_reps`) |
| `scores_out.npy` | Attack scores for out-world shadow models |
| `losses_in.npy` | Final training losses for in-world models |
| `losses_out.npy` | Final training losses for out-world models |
| `epsilon_lower.txt` | Computed empirical ε lower bound |
| `accuracy.json` | Clean and canary accuracy per shadow model |
| `metadata.json` | Full training hyperparameters |

## SLURM Scripts

Pre-configured job scripts in `scripts/` target TACC clusters (Stampede3, Frontera). Key experiments:

| Script | Experiment |
|---|---|
| `tc_mnist_blank.slurm` | Privacy-utility tradeoff curves for MNIST |
| `tc_cifar10_blank_cnn.slurm` | Privacy-utility tradeoff curves for CIFAR-10 |
| `gradient_bandwidth_attack.slurm` | Gradient bandwidth attack on MNIST |
| `defense_aware_mnist.slurm` | Defense-aware bilevel canary optimization |
| `defense_aware_mnist_audit.slurm` | Audit using defense-aware canaries |
| `cifar10_mislabeled_multi_canary.slurm` | Multi-canary audit on CIFAR-10 |
| `mnist_1000_reps.slurm` | High-statistical-power audit (1000 reps) |
| `fairness_audit.slurm` | Fairness impact analysis |

Submit with `sbatch scripts/<script>.slurm` or adapt for interactive use with `idev`.

## Algorithm

For each shadow model pair:
1. **In-world:** Train on `D ∪ {canary}`
2. **Out-world:** Train on `D` (no canary)
3. **Score:** Compute membership score (e.g., final loss, per-sample gradient norm)
4. **Attack:** Threshold scores to classify membership
5. **Evaluate:** Aggregate across all reps to compute empirical ε via GDP or conservative Poisson accounting

The resulting lower bound on ε certifies that the audited model leaks at least that much information about the canary.
