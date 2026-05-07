"""
Mini-test for the filter defense in parallel_audit_model_hamp.py.

Trains one in-world model with defense_type='none' and one with 'filter'
on a mislabeled canary, then checks that:
  1. Canary CE loss is HIGHER under the filter model (gradient ascent worked)
  2. Binary correctness on 18 augmentations is LOWER under the filter model
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils.data import load_data
from utils.training import xavier_init_model
from models import Models
from parallel_audit_model_hamp import (
    train_model,
    generate_augmentations,
    generate_binary_correctness_vector,
)


class Args:
    n_epochs = 20
    lr = 1e-4
    batch_size = 256
    block_size = None
    max_grad_norm = 1.0
    defense_k = 5
    defense_apply_ascent = True
    defense_score_fn = 'grad_norm'
    defense_score_norm = 'linf'
    defense_filter_every = 1
    # hyperparams required by DefenseConfig (defaults from build_parser)
    grad_norm_percentile_k = 20
    grad_dir_volatility_k = 5
    rand_proj_var_m = 10
    maxmin_proj_k = 10
    grad_rank_mode = 'effdim'
    grad_rank_eps = 1e-12
    dir_unique_k = 5
    alignment_proj_k = 10
    grad_scatter_k = 5
    loss_volatility_k = 5


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, y, out_dim = load_data('mnist', n_df=0)
    print(f"Loaded MNIST: {X.shape}")

    # Mislabeled canary: a real '3' image labeled as '1'
    # Mislabeling maximises gradient norm → easiest case for the defense to detect
    src_idx = np.where(y == 3)[0][0]
    canary_x = torch.from_numpy(X[src_idx]).float()
    canary_y = 1

    # In-world data: append canary at the last index
    X_in = np.vstack([X, canary_x.numpy()[np.newaxis]])
    y_in = np.concatenate([y, [canary_y]])
    print(f"In-world dataset: {X_in.shape}  (canary at index {len(X_in)-1})")

    # Shared initialisation
    torch.manual_seed(0)
    init_model = Models['cnn']((X.shape[1:], out_dim))
    xavier_init_model(init_model)

    args = Args()

    # ---- Train with no defense ----
    print("\nTraining: defense_type=none ...")
    model_none = copy.deepcopy(init_model).to(device)
    torch.manual_seed(1)
    train_model(model_none, X_in, y_in, canary_x, canary_y,
                device, args, defense_type='none')

    # ---- Train with filter defense ----
    print("Training: defense_type=filter ...")
    model_filter = copy.deepcopy(init_model).to(device)
    torch.manual_seed(1)
    train_model(model_filter, X_in, y_in, canary_x, canary_y,
                device, args, defense_type='filter')

    # ---- Compare canary CE loss ----
    model_none.eval()
    model_filter.eval()
    x_in = canary_x.unsqueeze(0).to(device)
    y_in_t = torch.tensor([canary_y], device=device)

    with torch.no_grad():
        loss_none = F.cross_entropy(model_none(x_in), y_in_t).item()
        loss_filter = F.cross_entropy(model_filter(x_in), y_in_t).item()

    print(f"\n--- Canary CE loss ---")
    print(f"  none:   {loss_none:.4f}")
    print(f"  filter: {loss_filter:.4f}")
    if loss_filter > loss_none:
        print("  PASS  filter loss > none (gradient ascent raised loss)")
    else:
        print("  FAIL  filter loss <= none")

    # ---- Compare binary correctness on 18 augmentations ----
    augmentations = generate_augmentations(canary_x, 18, use_flip=False)
    bv_none   = generate_binary_correctness_vector(model_none,   canary_x, canary_y, augmentations, device)
    bv_filter = generate_binary_correctness_vector(model_filter, canary_x, canary_y, augmentations, device)

    print(f"\n--- Binary correctness (18 augmentations) ---")
    print(f"  none:   {int(bv_none.sum())}/18  {bv_none.astype(int).tolist()}")
    print(f"  filter: {int(bv_filter.sum())}/18  {bv_filter.astype(int).tolist()}")
    if bv_filter.sum() < bv_none.sum():
        print("  PASS  filter correctness < none")
    elif bv_filter.sum() == bv_none.sum():
        print("  INCONCLUSIVE  same correctness (try more epochs or larger defense_k)")
    else:
        print("  FAIL  filter correctness > none")


if __name__ == '__main__':
    main()
