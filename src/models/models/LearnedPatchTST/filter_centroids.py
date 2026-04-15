# =============================================================================
# filter_centroids.py — Gini-adaptive centroid population filter for Phase 1
# =============================================================================
#
# PURPOSE:
#   After Phase 1 training, the saved meta.json contains active_centroid_ids
#   based on a soft marginal threshold.  This can include centroids with very
#   low hard-assignment population (near-zero timesteps).  This script:
#
#     1. Recomputes mean timestep population per active centroid over the full
#        training set using hard assignments.
#     2. Computes the Gini coefficient of the population distribution.
#     3. Derives the bottom-drop percentage adaptively:
#            drop_pct = clip(gini * 50,  min=5,  max=50)
#        — Even distribution (Gini~0) → drop ~5% (only the near-zeros)
#        — Collapsed distribution (Gini~0.9) → drop ~45%
#     4. Removes the bottom drop_pct centroids by population.
#     5. Prints a full diagnostic report.
#     6. Overwrites meta.json with updated effective_k and active_centroid_ids
#        (unless --dry_run is set).
#
# USAGE:
#   python filter_centroids.py \
#     --meta_path    saved_models/Wearable/learned_patchtst/LearnedPatch_phase1/Wearable_phase1_cluster_cw10080_K150_epochs30_model1_meta.json \
#     --patch_learner_path saved_models/Wearable/learned_patchtst/LearnedPatch_phase1/Wearable_phase1_cluster_cw10080_K150_epochs30_model1_patch_learner.pth \
#     --dset_finetune Wearable \
#     --context_points 10080 \
#     --n_patches 150 \
#     [--dry_run]
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np
import torch

_PATCHTST_DIR = os.path.join(os.path.dirname(__file__), '..', 'PatchTST_self_supervised')
sys.path.insert(0, os.path.abspath(_PATCHTST_DIR))

from datautils import get_dls
from src.models.layers.revin import RevIN
from patch_learner import SoftKMeansPatchLearner


# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(description='Gini-adaptive centroid filter')

parser.add_argument('--meta_path', type=str, required=True,
                    help='Path to *_meta.json saved by Phase 1')
parser.add_argument('--patch_learner_path', type=str, required=True,
                    help='Path to *_patch_learner.pth saved by Phase 1')

# Data — must match Phase 1 exactly
parser.add_argument('--dset_finetune', type=str, default='Wearable')
parser.add_argument('--context_points', type=int, default=10080)
parser.add_argument('--target_points', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--scaler', type=str, default='standard')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--revin', type=int, default=1)

# Model — must match Phase 1 exactly
parser.add_argument('--n_patches', type=int, default=150,
                    help='Total K used in Phase 1 (before filtering)')

# Filter control
parser.add_argument('--dry_run', action='store_true',
                    help='Print report only; do not overwrite meta.json')
parser.add_argument('--min_drop_pct', type=float, default=5.0,
                    help='Minimum bottom-drop percentage (default 5)')
parser.add_argument('--max_drop_pct', type=float, default=50.0,
                    help='Maximum bottom-drop percentage (default 50)')
parser.add_argument('--gini_scale', type=float, default=50.0,
                    help='drop_pct = clip(gini * gini_scale, min, max). Default 50.')

args = parser.parse_args()
args.dset = args.dset_finetune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Gini coefficient
# =============================================================================
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a 1-D array of non-negative values.

    Gini = 0  →  perfectly equal distribution
    Gini = 1  →  one element holds everything

    Uses the standard sorted-cumsum formula:
        G = (2 * sum(rank * value)) / (n * total) - (n+1)/n
    """
    values = np.array(values, dtype=float)
    values = np.clip(values, 0, None)           # ensure non-negative
    if values.sum() == 0:
        return 0.0
    values = np.sort(values)                    # ascending
    n = len(values)
    ranks = np.arange(1, n + 1)
    return float((2.0 * (ranks * values).sum()) / (n * values.sum()) - (n + 1) / n)


# =============================================================================
# Main
# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # Step 1: Load meta.json
    # -------------------------------------------------------------------------
    print('=' * 65)
    print('  CENTROID POPULATION FILTER — Gini-adaptive')
    print('=' * 65)
    print(f'\nLoading meta: {args.meta_path}')

    with open(args.meta_path, 'r') as f:
        meta = json.load(f)

    active_ids_original = meta['active_centroid_ids']
    effective_k_original = meta['effective_k']

    print(f'  effective_k from meta  : {effective_k_original}')
    print(f'  n_patches total        : {meta["n_patches_total"]}')
    print(f'  best_epoch             : {meta["best_epoch"]}')
    print(f'  best_val_loss          : {meta["best_val_loss"]:.6f}')

    # -------------------------------------------------------------------------
    # Step 2: Load patch learner
    # -------------------------------------------------------------------------
    print(f'\nLoading patch learner: {args.patch_learner_path}')

    # Infer n_features from saved centroids shape before instantiating
    state = torch.load(args.patch_learner_path, map_location='cpu')
    n_features = state['centroids'].shape[1]
    print(f'  n_features inferred    : {n_features}')
    print(f'  n_patches loaded       : {state["centroids"].shape[0]}')

    patch_learner = SoftKMeansPatchLearner(
        n_features=n_features,
        n_patches=args.n_patches,
        temperature=0.1,   # sharp but not zero (avoids numerical issues)
    )
    patch_learner.load_state_dict(state)
    patch_learner.to(device)
    patch_learner.eval()

    # -------------------------------------------------------------------------
    # Step 3: Load data (training split only — same as Phase 1 diagnostics)
    # -------------------------------------------------------------------------
    print('\nLoading training data...')
    dls = get_dls(args)

    revin = RevIN(n_features).to(device) if args.revin else None

    # -------------------------------------------------------------------------
    # Step 4: Recompute mean timestep population per active centroid
    #         over the full training set (hard assignments restricted to
    #         active_ids_original — consistent with Phase 1 diagnostics)
    # -------------------------------------------------------------------------
    print('Computing hard-assignment populations over training set...')

    K_eff = effective_k_original
    size_accum = torch.zeros(K_eff)
    n_samples = 0

    with torch.no_grad():
        for raw_batch in dls.train:
            if isinstance(raw_batch, dict):
                xb = raw_batch['inputs_embeds']
            else:
                xb = raw_batch[0]

            xb = xb.to(device)
            if revin is not None:
                xb = revin(xb, mode='norm')

            # patch_ids: [B, T] values in [0, K_eff-1]
            patch_ids = patch_learner.hard_assignments(
                xb, active_centroid_ids=active_ids_original
            )
            B, T = patch_ids.shape

            counts = torch.zeros(B, K_eff, device=device)
            counts.scatter_add_(1, patch_ids,
                                 torch.ones(B, T, device=device))

            size_accum += counts.cpu().sum(dim=0)
            n_samples += B

    mean_sizes = (size_accum / n_samples).numpy()   # [K_eff] mean timesteps/centroid/sample

    # -------------------------------------------------------------------------
    # Step 5: Gini coefficient → adaptive drop percentage
    # -------------------------------------------------------------------------
    gini = gini_coefficient(mean_sizes)
    raw_drop_pct = gini * args.gini_scale
    drop_pct = float(np.clip(raw_drop_pct, args.min_drop_pct, args.max_drop_pct))
    n_to_drop = int(np.floor(K_eff * drop_pct / 100.0))
    n_to_keep = K_eff - n_to_drop

    # -------------------------------------------------------------------------
    # Step 6: Sort by population, identify kept / dropped centroids
    # -------------------------------------------------------------------------
    sorted_idx = np.argsort(mean_sizes)           # ascending (smallest first)
    drop_local_idx  = sorted_idx[:n_to_drop]      # indices into active_ids_original
    keep_local_idx  = sorted_idx[n_to_drop:]      # indices into active_ids_original (largest)

    # Map back to original centroid IDs
    active_ids_array = np.array(active_ids_original)
    kept_orig_ids   = sorted(active_ids_array[keep_local_idx].tolist())
    dropped_orig_ids = sorted(active_ids_array[drop_local_idx].tolist())

    kept_sizes   = mean_sizes[keep_local_idx]
    dropped_sizes = mean_sizes[drop_local_idx]

    # -------------------------------------------------------------------------
    # Step 7: Print full diagnostic report
    # -------------------------------------------------------------------------
    print()
    print('=' * 65)
    print('  POPULATION DISTRIBUTION — BEFORE FILTERING')
    print('=' * 65)
    print(f'  Active centroids (from meta)     : {K_eff}')
    print(f'  Total samples evaluated          : {n_samples}')
    print(f'  Context window T                 : {args.context_points}')
    print(f'  Expected uniform size            : {args.context_points / K_eff:.1f}  (T / K_eff)')
    print()
    print(f'  Mean  timesteps/centroid/sample  : {mean_sizes.mean():.1f}')
    print(f'  Std   timesteps/centroid/sample  : {mean_sizes.std():.1f}')
    print(f'  Median                           : {np.median(mean_sizes):.1f}')
    print(f'  Min                              : {mean_sizes.min():.1f}')
    print(f'  Max                              : {mean_sizes.max():.1f}')
    print(f'  Imbalance ratio (max/min)        : {mean_sizes.max() / (mean_sizes.min() + 1e-8):.1f}x')

    print()
    print('  Population histogram (10 buckets):')
    counts_hist, edges = np.histogram(mean_sizes, bins=10)
    for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts_hist)):
        bar = '#' * int(c * 30 / max(int(counts_hist.max()), 1))
        print(f'    [{lo:7.1f} – {hi:7.1f}]  {c:4d} centroids  {bar}')

    print()
    print('=' * 65)
    print('  GINI-ADAPTIVE DROP DECISION')
    print('=' * 65)
    print(f'  Gini coefficient                 : {gini:.4f}')
    print(f'    (0 = perfectly even, 1 = one centroid owns everything)')
    print()
    print(f'  Formula: drop_pct = clip(Gini × {args.gini_scale:.0f},  '
          f'min={args.min_drop_pct:.0f}%,  max={args.max_drop_pct:.0f}%)')
    print(f'  Raw drop pct (Gini × scale)      : {raw_drop_pct:.1f}%')
    print(f'  Clamped drop pct                 : {drop_pct:.1f}%')
    print(f'  Centroids to drop                : {n_to_drop}  '
          f'(bottom {drop_pct:.1f}% by population)')
    print(f'  Centroids to keep                : {n_to_keep}')

    print()
    print('  Top 10 kept centroids (largest → smallest):')
    top10 = np.argsort(kept_sizes)[::-1][:10]
    for rank, li in enumerate(top10):
        orig = active_ids_array[keep_local_idx[li]]
        print(f'    #{rank+1:2d}: orig_centroid[{orig:3d}]  →  {kept_sizes[li]:.1f} timesteps/sample')

    print()
    print('  Bottom 10 dropped centroids (largest → smallest within dropped):')
    bot10 = np.argsort(dropped_sizes)[::-1][:10]
    for rank, li in enumerate(bot10):
        orig = active_ids_array[drop_local_idx[li]]
        print(f'    #{rank+1:2d}: orig_centroid[{orig:3d}]  →  {dropped_sizes[li]:.1f} timesteps/sample  ← REMOVED')

    print()
    print('=' * 65)
    print('  AFTER FILTERING')
    print('=' * 65)
    print(f'  effective_k (new)                : {n_to_keep}  (was {K_eff})')
    print(f'  active_centroid_ids (first 20)   : '
          f'{kept_orig_ids[:20]}{"..." if len(kept_orig_ids) > 20 else ""}')
    print()

    new_mean = kept_sizes.mean()
    new_std  = kept_sizes.std()
    new_med  = float(np.median(kept_sizes))
    new_min  = kept_sizes.min()
    new_max  = kept_sizes.max()
    new_gini = gini_coefficient(kept_sizes)
    new_imbalance = new_max / (new_min + 1e-8)

    print(f'  Mean  timesteps/centroid/sample  : {new_mean:.1f}')
    print(f'  Std   timesteps/centroid/sample  : {new_std:.1f}')
    print(f'  Median                           : {new_med:.1f}')
    print(f'  Min                              : {new_min:.1f}')
    print(f'  Max                              : {new_max:.1f}')
    print(f'  Imbalance ratio (max/min)        : {new_imbalance:.1f}x  (was {mean_sizes.max() / (mean_sizes.min() + 1e-8):.1f}x)')
    print(f'  Gini (after)                     : {new_gini:.4f}  (was {gini:.4f})')
    print()
    print(f'  Expected uniform patch size      : {args.context_points / n_to_keep:.1f}  (T / K_new)')

    # -------------------------------------------------------------------------
    # Step 8: Write updated meta.json (unless --dry_run)
    # -------------------------------------------------------------------------
    if args.dry_run:
        print()
        print('  DRY RUN — meta.json NOT updated.')
        print('  Remove --dry_run to write changes.')
    else:
        meta_new = dict(meta)
        meta_new['effective_k']          = n_to_keep
        meta_new['active_centroid_ids']  = kept_orig_ids
        meta_new['gini_before_filter']   = round(gini, 6)
        meta_new['gini_after_filter']    = round(float(new_gini), 6)
        meta_new['drop_pct_applied']     = round(drop_pct, 2)
        meta_new['n_dropped']            = n_to_drop
        meta_new['dropped_centroid_ids'] = dropped_orig_ids

        with open(args.meta_path, 'w') as f:
            json.dump(meta_new, f, indent=2)

        print(f'  meta.json updated: {args.meta_path}')

    print('=' * 65)


if __name__ == '__main__':
    main()
