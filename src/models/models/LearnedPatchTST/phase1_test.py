# =============================================================================
# phase1_test.py — Test a saved Phase 1 PatchLearnerClassifier on the test set
# =============================================================================
#
# Loads a saved Phase 1 checkpoint (PatchLearnerClassifier = clustering + linear
# head) and evaluates it on the held-out test set using the same metrics and
# threshold selection logic as patchtst_finetune.py:
#
#   Val set  → Youden's J threshold selection (no leakage)
#   Test set → all metrics applied blind with val-derived threshold
#
#   Metrics reported:
#     PR-AUC, ROC-AUC  (threshold-independent)
#     @0.5:   TP/FP/TN/FN, Sensitivity, Specificity, Precision, NPV,
#             F1, F2, MCC, Balanced Accuracy
#     @opt:   same metrics at Youden's J threshold from val set
#
# USAGE:
#   python phase1_test.py \
#     --model_path saved_models/.../Wearable_phase1_cluster_cw10080_K150_epochs30_model3.pth \
#     --dset_finetune Wearable \
#     --context_points 10080 \
#     --revin 1 \
#     --use_pos_enc 1
#
# NOTE: n_patches and balancing_lambda are inferred from the checkpoint —
# do not pass them. Only pass flags that affect data loading or preprocessing.
# =============================================================================

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve,
)

_PATCHTST_DIR = os.path.join(os.path.dirname(__file__), '..', 'PatchTST_self_supervised')
sys.path.insert(0, os.path.abspath(_PATCHTST_DIR))

from datautils import get_dls
from src.models.layers.revin import RevIN
from patch_learner import PatchLearnerClassifier, sinusoidal_pos_enc


# =============================================================================
# Arguments — must match Phase 1 training args exactly
# =============================================================================
parser = argparse.ArgumentParser(description='Phase 1 test evaluation')

parser.add_argument('--model_path', type=str, required=True,
                    help='path to saved Phase 1 .pth checkpoint')

# Data
parser.add_argument('--dset_finetune', type=str, default='Wearable')
parser.add_argument('--context_points', type=int, default=10080)
parser.add_argument('--target_points', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--scaler', type=str, default='standard')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--revin', type=int, default=1)

# balancing_lambda is a training-only regularizer — not needed at test time
parser.add_argument('--use_pos_enc', type=int, default=1)

args = parser.parse_args()
args.dset = args.dset_finetune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Metric helpers — identical to patchtst_finetune.py
# =============================================================================
def _metrics_at_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    f2          = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    mcc         = matthews_corrcoef(y_true, y_pred)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)

    return dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
        f1=f1,
        f2=f2,
        mcc=mcc,
        balanced_accuracy=bal_acc,
    )


def _interpret_sensitivity(s, n_pos):
    caught, missed = round(s * n_pos), n_pos - round(s * n_pos)
    if s == 0.0:
        return 'catching NO flu cases — model predicts all negative at this threshold'
    elif s < 0.3:
        return f'catching only {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — very poor'
    elif s < 0.6:
        return f'catching {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — moderate'
    elif s < 0.8:
        return f'catching {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — reasonable'
    else:
        return f'catching {caught}/{n_pos} flu cases — missing only {missed} ({(1-s)*100:.0f}%) — strong'


def _interpret_mcc(mcc):
    if mcc <= 0.0:   return 'no useful signal (≤ random)'
    elif mcc < 0.2:  return 'weak signal'
    elif mcc < 0.4:  return 'moderate signal'
    elif mcc < 0.6:  return 'good signal'
    else:            return 'strong signal'


def _interpret_pr_auc(pr_auc, prevalence):
    if prevalence <= 0: return ''
    lift = pr_auc / prevalence
    if lift < 2:    return f'{lift:.1f}x above random — poor'
    elif lift < 10: return f'{lift:.1f}x above random — moderate'
    elif lift < 50: return f'{lift:.1f}x above random — good'
    else:           return f'{lift:.1f}x above random — excellent'


def _print_metrics_block(label, m, threshold, n_pos):
    print(f'  {"─"*60}')
    print(f'  Threshold : {threshold:.4f}  ({label})')
    print(f'  {"─"*60}')
    print(f'  Confusion matrix:')
    print(f'    TP (flu caught)      = {m["TP"]:6d}')
    print(f'    FN (flu missed)      = {m["FN"]:6d}   ← minimise this')
    print(f'    FP (false alarms)    = {m["FP"]:6d}')
    print(f'    TN (healthy correct) = {m["TN"]:6d}')
    print(f'  {"─"*60}')
    print(f'  Sensitivity (Recall)  : {m["sensitivity"]:.4f}  — {_interpret_sensitivity(m["sensitivity"], n_pos)}')
    print(f'  Specificity           : {m["specificity"]:.4f}  — fraction of healthy windows correctly ID\'d')
    print(f'  Precision (PPV)       : {m["precision"]:.4f}  — of flagged windows, fraction truly flu')
    print(f'  NPV                   : {m["npv"]:.4f}  — of cleared windows, fraction truly healthy')
    print(f'  {"─"*60}')
    print(f'  F1                    : {m["f1"]:.4f}  — balanced precision/recall')
    print(f'  F2 (recall-weighted)  : {m["f2"]:.4f}  — weights missing a case 2x worse than false alarm')
    print(f'  MCC                   : {m["mcc"]:.4f}  — {_interpret_mcc(m["mcc"])}')
    print(f'  Balanced Accuracy     : {m["balanced_accuracy"]:.4f}  — accuracy corrected for imbalance (0.5=random)')


# =============================================================================
# Inference helper — runs one dataloader split through the model
# =============================================================================
def _run_inference(loader, model, revin, pos_enc):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for raw_batch in loader:
            if isinstance(raw_batch, dict):
                xb, yb = raw_batch['inputs_embeds'], raw_batch['label']
            else:
                xb, yb = raw_batch

            xb = xb.to(device)
            yb = yb.to(device).float()

            if revin is not None:
                xb = revin(xb, mode='norm')
            if pos_enc is not None:
                xb = xb + pos_enc.unsqueeze(0)

            logits, _ = model(xb)
            all_logits.append(logits.squeeze(-1).cpu())
            all_labels.append(yb.cpu())

    logits = torch.cat(all_logits).numpy().reshape(-1)
    labels = torch.cat(all_labels).numpy().reshape(-1).astype(int)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels


# =============================================================================
# Main
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # Load all splits — val for threshold, test for evaluation
    # -------------------------------------------------------------------------
    dls        = get_dls(args)
    n_features = dls.vars

    print(f'n_features  : {n_features}')
    print(f'val  labels : {dls.valid.dataset.data["label"].unique(return_counts=True)}')
    print(f'test labels : {dls.test.dataset.data["label"].unique(return_counts=True)}')

    # -------------------------------------------------------------------------
    # Build model and load weights
    # Infer n_patches from the saved checkpoint — no need to pass it as an arg.
    # The full K (e.g. 150) is always used here, matching training exactly.
    # Dead centroid filtering only applies to Phase 2 onwards (via meta.json).
    # -------------------------------------------------------------------------
    state      = torch.load(args.model_path, map_location=device)
    n_patches  = state['patch_learner.centroids'].shape[0]
    print(f'n_patches   : {n_patches}  (inferred from checkpoint)')

    model = PatchLearnerClassifier(
        n_features=n_features,
        n_patches=n_patches,
        temperature=0.1,
        balancing_lambda=0.0,   # not used at test time — no loss computed
        n_classes=1,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    print(f'Weights loaded from: {args.model_path}')

    # -------------------------------------------------------------------------
    # RevIN and positional encoding — must match Phase 1 training exactly
    # -------------------------------------------------------------------------
    revin   = RevIN(n_features).to(device) if args.revin else None
    pos_enc = sinusoidal_pos_enc(args.context_points, n_features, device) \
              if args.use_pos_enc else None

    # -------------------------------------------------------------------------
    # Step 1 — Val inference: find Youden's J threshold
    # Find threshold on val, apply blind to test — no leakage.
    # -------------------------------------------------------------------------
    val_probs, val_true = _run_inference(dls.valid, model, revin, pos_enc)

    if len(np.unique(val_true)) >= 2:
        fpr_arr, tpr_arr, thresh_arr = roc_curve(val_true, val_probs)
        youden_idx    = int(np.argmax(tpr_arr - fpr_arr))
        opt_threshold = float(thresh_arr[youden_idx])
        val_j         = float(tpr_arr[youden_idx] - fpr_arr[youden_idx])
    else:
        opt_threshold = 0.5
        val_j         = float('nan')

    # -------------------------------------------------------------------------
    # Step 2 — Test inference: evaluate blind
    # -------------------------------------------------------------------------
    probs, y_true = _run_inference(dls.test, model, revin, pos_enc)

    prevalence = y_true.mean()
    n_pos      = int(y_true.sum())

    # Threshold-independent
    if len(np.unique(y_true)) >= 2:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc  = average_precision_score(y_true, probs)
    else:
        roc_auc = pr_auc = float('nan')

    # Threshold-dependent
    m_05  = _metrics_at_threshold(y_true, probs, threshold=0.5)
    m_opt = _metrics_at_threshold(y_true, probs, threshold=opt_threshold) \
            if not np.isnan(opt_threshold) else None

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------
    print()
    print(f'  {"="*60}')
    print(f'  PHASE 1 TEST RESULTS')
    print(f'  {"="*60}')
    print(f'  NOTE: with {prevalence*100:.3f}% prevalence, accuracy and ROC-AUC are')
    print(f'  misleading. Focus on PR-AUC, MCC, Sensitivity, and F2.')
    print(f'  The @optimal block is more informative than @0.5 because')
    print(f'  @0.5 almost always predicts all-negative at this prevalence.')
    print(f'  {"="*60}')
    print(f'  Dataset (test set):')
    print(f'    Total samples  : {len(y_true)}')
    print(f'    Positives      : {n_pos}  (flu-onset windows)')
    print(f'    Negatives      : {len(y_true) - n_pos}')
    print(f'    Prevalence     : {prevalence:.5f}  ({prevalence*100:.3f}%)')
    print(f'    Random PR-AUC  : {prevalence:.5f}  (baseline)')
    print()
    print(f'  Threshold selection (on VALIDATION set — not test):')
    print(f'    Youden\'s J threshold : {opt_threshold:.4f}')
    print(f'    J statistic (val)    : {val_j:.4f}  (0=random, 1=perfect)')
    print(f'    Val positives        : {val_true.sum()} / {len(val_true)}')
    print()
    print(f'  Threshold-independent metrics (test set):')
    print(f'    PR-AUC  (primary) : {pr_auc:.4f}  — {_interpret_pr_auc(pr_auc, prevalence)}')
    print(f'    ROC-AUC           : {roc_auc:.4f}  — ranking ability (0.5=random, 1.0=perfect)')
    print()
    _print_metrics_block('@0.5  (fixed reference)', m_05, threshold=0.5, n_pos=n_pos)
    if m_opt is not None:
        print()
        _print_metrics_block(
            "optimal Youden's J  (threshold from VAL, applied to TEST)",
            m_opt, threshold=opt_threshold, n_pos=n_pos
        )
    print(f'  {"="*60}')


if __name__ == '__main__':
    main()
