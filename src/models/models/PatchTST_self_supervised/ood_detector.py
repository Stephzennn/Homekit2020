"""
ood_detector.py — Out-of-distribution detection via Mahalanobis distance
on PatchTST backbone embeddings.

Same embedding extraction as plot_embeddings.py (backbone → last patch →
flatten → [B x nvars*d_model]) but skips t-SNE and feeds the full
high-dimensional vector into a robust covariance estimator (LedoitWolf).

Supports 1 or 2 pretrained models:
  - 1 model  : single in-distribution reference (e.g. positive-only trained)
  - 2 models : two references (positive + negative trained); score = min distance
               across both, meaning "how far is this from the nearest known
               distribution?"

Modes
-----
  fit      — extract embeddings from training data, fit covariance, save detector
  evaluate — load saved detector, score val/test split, print AUROC / threshold

Usage (run from the PatchTST_self_supervised directory)
--------------------------------------------------------
# Fit on positive-only model:
python ood_detector.py --mode fit \
    --model_paths /path/to/positive.pth \
    --save_detector ood_positive.pkl \
    --c_in 8 --d_model 256 --n_layers 4 --n_heads 8 --d_ff 512 \
    --patch_len 1440 --stride 180 --context_points 10080

# Fit on both positive + negative models:
python ood_detector.py --mode fit \
    --model_paths /path/to/positive.pth /path/to/negative.pth \
    --save_detector ood_both.pkl \
    ...

# Evaluate saved detector on test set:
python ood_detector.py --mode evaluate \
    --load_detector ood_both.pkl \
    --threshold 35.0 \
    ...
"""

import argparse
import os
import sys
import pickle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

_PROJECT_ROOT = "/home/hice1/ezg6/projects/Homekit2020/src"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

from src.models.patchTST import PatchTST
from src.models.layers.revin import RevIN
from src.callback.patch_mask import create_patch
from datautils import get_dls


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="OOD detection via Mahalanobis distance on PatchTST embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode", type=str, required=True, choices=["fit", "evaluate"],
                   help="'fit' extracts embeddings and saves detector; 'evaluate' scores a split")

    # --- Model checkpoints (1 or 2) ---
    p.add_argument("--model_paths", nargs="+", required=True,
                   help="Path(s) to pretrained .pth checkpoint(s). 1 = single reference, "
                        "2 = positive+negative (score = min distance across both)")
    p.add_argument("--save_detector", type=str, default="ood_detector.pkl",
                   help="Where to save the fitted detector (fit mode)")
    p.add_argument("--load_detector", type=str, default="ood_detector.pkl",
                   help="Path to a previously saved detector (evaluate mode)")

    # --- Architecture (must match checkpoints) ---
    p.add_argument("--c_in",         type=int,   default=8,    help="Number of input channels")
    p.add_argument("--d_model",      type=int,   default=256,  help="Transformer d_model")
    p.add_argument("--n_layers",     type=int,   default=4,    help="Number of Transformer layers")
    p.add_argument("--n_heads",      type=int,   default=8,    help="Number of attention heads")
    p.add_argument("--d_ff",         type=int,   default=512,  help="Transformer FFN hidden dim")
    p.add_argument("--dropout",      type=float, default=0.1,  help="Transformer dropout")
    p.add_argument("--head_dropout", type=float, default=0.1,  help="Head dropout")

    # --- Patching ---
    p.add_argument("--patch_len",      type=int, default=1440,  help="Patch length")
    p.add_argument("--stride",         type=int, default=180,   help="Stride between patches")
    p.add_argument("--context_points", type=int, default=10080, help="Sequence length")

    # --- Data ---
    p.add_argument("--batch_size",  type=int, default=32, help="Batch size for inference")
    p.add_argument("--num_workers", type=int, default=0,  help="DataLoader workers")
    p.add_argument("--revin",       type=int, default=1,  help="Use RevIN normalization")

    # --- Evaluate mode ---
    p.add_argument("--threshold", type=float, default=None,
                   help="Mahalanobis distance threshold for OOD decision. "
                        "If not set, the 95th percentile of training scores is used.")
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"],
                   help="Which split to evaluate in evaluate mode")
    p.add_argument("--out_dir", type=str, default=".",
                   help="Directory to save output plots")
    p.add_argument("--combine", type=str, default="bayes",
                   choices=["min", "bayes", "average"],
                   help=(
                       "How to combine scores from two models. "
                       "'bayes'  = sqrt(d_pos²+d_neg²)  — product of likelihoods, strict; "
                       "'min'    = min(d_pos, d_neg)    — nearest distribution, lenient; "
                       "'average'= (d_pos+d_neg)/2      — arithmetic mean"
                   ))

    return p


# ---------------------------------------------------------------------------
# Model construction — identical to plot_embeddings.py
# ---------------------------------------------------------------------------
def build_model(checkpoint_path, args, device):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1

    model = PatchTST(
        c_in=args.c_in,
        target_dim=args.patch_len,   # pretrain head shape
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act="relu",
        head_type="pretrain",
        res_attention=False,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    print(f"Loaded: {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Embedding hook — same as plot_embeddings.py pretrain mode:
# backbone output [B x nvars x d_model x num_patch]
#   → last patch  [:, :, :, -1]  → [B x nvars x d_model]
#   → flatten                    → [B x nvars * d_model]
# ---------------------------------------------------------------------------
def register_embedding_hook(model):
    """
    Captures backbone output as a fixed-size vector per sample.

    backbone → [B x nvars x d_model x num_patch]
      mean over patches → [B x nvars x d_model]
      reshape           → [B x nvars * d_model]

    Mean pooling over all patches gives a more stable representation than
    taking only the last patch, and is the natural ID centroid for Mahalanobis.
    """
    captured = []

    def _pre_hook(module, inp):
        x = inp[0].detach().cpu()               # [B x nvars x d_model x num_patch]
        x = x.mean(dim=-1)                      # [B x nvars x d_model]
        x = x.reshape(x.shape[0], -1)           # [B x nvars * d_model]
        captured.append(x)

    handle = model.head.register_forward_pre_hook(_pre_hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, dataloader, revin, captured, args, device):
    all_emb = []
    all_lbl = []

    for batch in dataloader:
        if isinstance(batch, dict):
            xb = batch["inputs_embeds"].float().to(device)
            yb = batch["label"].float()
        elif isinstance(batch, (list, tuple)):
            xb = batch[0].float().to(device)
            yb = batch[1].float()
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if args.revin:
            xb = revin(xb, "norm")

        xb_patch, _ = create_patch(xb, args.patch_len, args.stride)
        _ = model(xb_patch)

        all_emb.append(captured.pop())
        all_lbl.append(yb.reshape(-1).cpu())

    return (
        torch.cat(all_emb, dim=0).numpy(),
        torch.cat(all_lbl, dim=0).numpy(),
    )


# ---------------------------------------------------------------------------
# OOD Detector
# ---------------------------------------------------------------------------
class MahalanobisOODDetector:
    """
    Fits a LedoitWolf robust covariance estimator on in-distribution embeddings
    from one or two pretrained PatchTST models.

    Scoring modes (--combine)
    -------------------------
    min    : score = min(d_pos, d_neg)
             "How far is x from the NEAREST known distribution?"
             Lenient: only ONE model needs to claim x as in-distribution.

    bayes  : score = sqrt(d_pos² + d_neg²)
             Derived from multiplying the Gaussian likelihoods of both models:
               log p(x|both) ∝ -½d_pos² + (-½d_neg²)
             OOD score = -log p(x|both) ∝ d_pos² + d_neg²  → take sqrt for scale.
             Strict: BOTH models must agree x is in-distribution.

    average: score = (d_pos + d_neg) / 2
             Simple arithmetic mean — less principled than bayes but intuitive.

    Class probability (two-model only)
    -----------------------------------
    Given both distances, we can also ask "which distribution does x resemble?"
    using a Bayesian posterior over the two Gaussians:

        log_pos = -½ * d_pos²      (log-likelihood under positive model)
        log_neg = -½ * d_neg²      (log-likelihood under negative model)
        p(positive | x) = softmax([log_pos, log_neg])[0]
                        = exp(log_pos) / (exp(log_pos) + exp(log_neg))
                        = sigmoid(½ * (d_neg² - d_pos²))

    Returns a value in [0, 1]:
        → 1.0  : x strongly resembles the positive (flu-onset) distribution
        → 0.0  : x strongly resembles the negative (healthy) distribution
        → 0.5  : ambiguous — equidistant from both, or both distances large (OOD)
    """

    def __init__(self, combine: str = "bayes"):
        assert combine in ("min", "bayes", "average"), \
            "combine must be 'min', 'bayes', or 'average'"
        self.combine = combine
        self.estimators = []
        self.train_scores = None
        self.threshold = None
        self.embedding_dim = None

    def fit(self, embeddings_list):
        """
        Parameters
        ----------
        embeddings_list : list of np.ndarray, shape (N, d) each
            One array per model (positive, then negative if two models).
        """
        self.estimators = []

        for i, emb in enumerate(embeddings_list):
            label = ["positive", "negative"][i] if len(embeddings_list) == 2 else str(i)
            print(f"  Fitting LedoitWolf on {label} model embeddings "
                  f"({emb.shape[0]} samples, dim={emb.shape[1]}) …")
            lw = LedoitWolf(assume_centered=False)
            lw.fit(emb)
            self.estimators.append(lw)
            print(f"    Shrinkage coefficient: {lw.shrinkage_:.4f}")

        self.embedding_dim = embeddings_list[0].shape[1]
        self.train_scores = self._combine_distances(
            [self._mahalanobis(lw, emb)
             for lw, emb in zip(self.estimators, embeddings_list)]
        )

        print(f"\n  Training score distribution (combine='{self.combine}'):")
        for p in (50, 90, 95, 99):
            print(f"    p{p:02d} = {np.percentile(self.train_scores, p):.3f}")

        self.threshold = float(np.percentile(self.train_scores, 95))
        print(f"\n  Default threshold (95th percentile of train): {self.threshold:.4f}")

    # ------------------------------------------------------------------
    def _combine_distances(self, distance_list):
        """Combine per-model Mahalanobis distances into a single OOD score."""
        if len(distance_list) == 1:
            return distance_list[0]

        d_pos, d_neg = distance_list[0], distance_list[1]

        if self.combine == "min":
            return np.minimum(d_pos, d_neg)

        elif self.combine == "bayes":
            # Bayesian (product of likelihoods):
            # log p(x|both) ∝ -½(d_pos² + d_neg²)
            # OOD score ∝ sqrt(d_pos² + d_neg²)
            return np.sqrt(d_pos ** 2 + d_neg ** 2)

        elif self.combine == "average":
            return (d_pos + d_neg) / 2.0

    @staticmethod
    def _mahalanobis(lw, X):
        """Vectorised Mahalanobis: sqrt( (X-μ)ᵀ Σ⁻¹ (X-μ) ) for each row."""
        diff = X - lw.location_                             # [N, d]
        sq = np.einsum('nd,de,ne->n', diff, lw.precision_, diff)
        return np.sqrt(np.maximum(sq, 0.0))                 # [N]

    # ------------------------------------------------------------------
    def score(self, X):
        """
        OOD score for each sample.  Higher = more out-of-distribution.

        Returns
        -------
        scores : np.ndarray, shape (N,)
        """
        distances = [self._mahalanobis(lw, X) for lw in self.estimators]
        return self._combine_distances(distances)

    def class_probability(self, X):
        """
        Bayesian posterior P(flu | x) using the full Gaussian log-likelihood ratio.

        Derivation (equal priors P(flu) = P(healthy) = ½):
        ─────────────────────────────────────────────────────────────────────
        P(flu | x) = P(x | flu) / [P(x | flu) + P(x | healthy)]
                   = σ( log P(x|flu) − log P(x|healthy) )

        Each model is a multivariate Gaussian (LedoitWolf):
          log P(x | m) = −½ dₘ² − ½ log|Σₘ| − (k/2)log(2π)

        Log-likelihood ratio (2π term cancels):
          log P(x|flu) − log P(x|neg)
            = −½ d_flu² − ½ log|Σ_flu| − (−½ d_neg² − ½ log|Σ_neg|)
            = ½(d_neg² − d_flu²)          ← distance term
            + ½ log(|Σ_neg| / |Σ_flu|)   ← volume correction

        Volume correction: a wide distribution (large |Σ|) assigns higher
        likelihood to all points; without this correction a model trained on
        more varied data would always "win".

        Since LedoitWolf gives Precision = Σ⁻¹:
          log|Σ| = −log|Precision|  →  log(|Σ_neg|/|Σ_flu|)
                                       = log|Precision_flu| − log|Precision_neg|

        Full formula:
          P(flu | x) = σ( ½(d_neg² − d_flu²) + ½(log|Λ_flu| − log|Λ_neg|) )

        where Λ = Precision matrix (Σ⁻¹).
        ─────────────────────────────────────────────────────────────────────

        Returns
        -------
        p_positive : np.ndarray, shape (N,)  values in [0, 1]
            → 1.0 : x strongly resembles the flu-onset (positive) distribution
            → 0.0 : x strongly resembles the healthy (negative) distribution
            → 0.5 : ambiguous — equidistant or both distances large (OOD)
        """
        if len(self.estimators) != 2:
            raise RuntimeError("class_probability requires exactly 2 fitted models.")

        lw_flu, lw_neg = self.estimators[0], self.estimators[1]

        d_flu = self._mahalanobis(lw_flu, X)
        d_neg = self._mahalanobis(lw_neg, X)

        # Volume correction: ½ log(|Σ_neg|/|Σ_flu|) = ½(log|Λ_flu| − log|Λ_neg|)
        # np.linalg.slogdet returns (sign, log|M|)
        log_det_flu = np.linalg.slogdet(lw_flu.precision_)[1]
        log_det_neg = np.linalg.slogdet(lw_neg.precision_)[1]
        volume_correction = 0.5 * (log_det_flu - log_det_neg)

        log_ratio = 0.5 * (d_neg ** 2 - d_flu ** 2) + volume_correction
        return 1.0 / (1.0 + np.exp(-log_ratio))    # sigmoid

    def predict(self, X, threshold=None):
        """
        Returns
        -------
        is_ood       : bool array (N,)  True = out-of-distribution
        ood_score    : float array (N,)
        p_positive   : float array (N,) or None if single model
        """
        t = threshold if threshold is not None else self.threshold
        ood_score  = self.score(X)
        is_ood     = ood_score > t
        p_positive = self.class_probability(X) if len(self.estimators) == 2 else None
        return is_ood, ood_score, p_positive

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Detector saved → {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            det = pickle.load(f)
        print(f"Detector loaded ← {path}")
        return det


# ---------------------------------------------------------------------------
# Data loading shim — mirrors plot_embeddings.py
# ---------------------------------------------------------------------------
def make_data_args(args):
    class _Args:
        dset           = "Wearable"
        context_points = args.context_points
        target_points  = 1
        batch_size     = args.batch_size
        num_workers    = args.num_workers
        scaler         = "standard"
        features       = "M"
        patch_len      = args.patch_len
        stride         = args.stride
        revin          = args.revin
        label_filter   = "all"
    return _Args()


# ---------------------------------------------------------------------------
# Evaluate: score a split, compute AUROC, plot score distributions
# ---------------------------------------------------------------------------
def evaluate(detector, embeddings, labels, threshold, out_dir, split_name, model_tag):
    is_ood, scores, p_positive = detector.predict(embeddings, threshold)
    t = threshold if threshold is not None else detector.threshold

    n_classes = len(np.unique(labels))
    prevalence = labels.mean()
    if n_classes >= 2:
        pr_auc = average_precision_score(labels, scores)
        auroc  = roc_auc_score(labels, scores)
    else:
        pr_auc = auroc = float("nan")

    print(f"\n  ── {split_name.upper()} ──────────────────────────────────────")
    print(f"  Combine method   : {detector.combine}")
    print(f"  Threshold        : {t:.4f}")
    print(f"  Flagged as OOD   : {is_ood.sum()} / {len(is_ood)}  ({100*is_ood.mean():.1f}%)")
    print(f"  OOD score stats  : mean={scores.mean():.3f}  std={scores.std():.3f}  "
          f"p50={np.percentile(scores,50):.3f}  p95={np.percentile(scores,95):.3f}")
    if n_classes >= 2:
        print(f"  PR-AUC (primary) : {pr_auc:.4f}  "
              f"(random baseline = {prevalence:.4f}  |  lift = {pr_auc/prevalence:.1f}x)")
        print(f"  ROC-AUC          : {auroc:.4f}")

    if p_positive is not None:
        print(f"\n  Bayesian class probability  p(positive | x):")
        print(f"    mean over all samples : {p_positive.mean():.3f}")
        for lbl, name in [(0, "label=0 (healthy)"), (1, "label=1 (flu)")]:
            mask = labels == lbl
            if mask.sum() > 0:
                print(f"    {name:25s}: mean={p_positive[mask].mean():.3f}  "
                      f"std={p_positive[mask].std():.3f}")

    # ── Plots ────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    n_cols = 3 if p_positive is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # Panel 1: OOD score distribution
    ax = axes[0]
    for lbl, color, name in [(0, "#4878d0", "Label 0 (healthy)"),
                              (1, "#d65f5f", "Label 1 (flu)")]:
        mask = labels == lbl
        if mask.sum() > 0:
            ax.hist(scores[mask], bins=60, alpha=0.6, color=color,
                    label=f"{name}  (n={mask.sum()})", density=True)
    ax.axvline(t, color="black", lw=1.5, ls="--", label=f"threshold={t:.2f}")
    ax.set_xlabel("OOD score (Mahalanobis combined)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"OOD score distribution\n{split_name}  combine='{detector.combine}'", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: PR curve (primary for imbalanced data)
    ax = axes[1]
    if n_classes >= 2:
        prec, rec, _ = precision_recall_curve(labels, scores)
        ax.plot(rec, prec, lw=2, label=f"PR-AUC={pr_auc:.3f}")
        ax.axhline(prevalence, color="gray", lw=1, ls="--",
                   label=f"random baseline ({prevalence:.4f})")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "single class\nin split", ha="center", va="center")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"PR curve — {split_name}", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    # Panel 3: Bayesian class probability (two-model only)
    if p_positive is not None:
        ax = axes[2]
        for lbl, color, name in [(0, "#4878d0", "Label 0 (healthy)"),
                                  (1, "#d65f5f", "Label 1 (flu)")]:
            mask = labels == lbl
            if mask.sum() > 0:
                ax.hist(p_positive[mask], bins=50, alpha=0.6, color=color,
                        label=f"{name}  (n={mask.sum()})", density=True)
        ax.axvline(0.5, color="gray", lw=1, ls="--", label="p=0.5")
        ax.set_xlabel("p(positive | x)  — Bayesian class probability", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Bayesian class probability\n"
                     "high → flu-onset distribution  |  low → healthy distribution",
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"ood_{split_name}_{model_tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")

    return auroc, scores, p_positive


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    if len(args.model_paths) > 2:
        raise ValueError("--model_paths accepts at most 2 checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode:   {args.mode}")
    print(f"Models: {args.model_paths}\n")

    # Build one RevIN (shared architecture across models)
    revin = RevIN(num_features=args.c_in, eps=1e-5, affine=False).to(device)

    # Data
    data_args = make_data_args(args)
    data_args.dset = "Wearable"
    dls = get_dls(data_args)

    split_map = {"train": dls.train, "val": dls.valid, "test": dls.test}

    model_tag = f"{len(args.model_paths)}model"

    # ---- FIT MODE -----------------------------------------------------------
    if args.mode == "fit":
        print("=" * 60)
        print("FIT MODE — extracting embeddings + fitting covariance")
        print("=" * 60)

        all_embeddings = []

        for ckpt in args.model_paths:
            print(f"\nModel: {ckpt}")
            model = build_model(ckpt, args, device)
            captured, hook = register_embedding_hook(model)

            dl = split_map["train"]
            emb, lbl = extract_embeddings(model, dl, revin, captured, args, device)
            hook.remove()

            print(f"  Extracted {emb.shape[0]} embeddings, dim={emb.shape[1]}")
            print(f"  Label distribution: "
                  f"{int((lbl==1).sum())} positive, {int((lbl==0).sum())} negative")
            all_embeddings.append(emb)

        print("\nFitting MahalanobisOODDetector …")
        detector = MahalanobisOODDetector(combine=args.combine)
        detector.fit(all_embeddings)
        detector.save(args.save_detector)

        # Quick self-evaluation on training data
        print("\nSelf-evaluation on training data:")
        train_emb = all_embeddings[0] if len(all_embeddings) == 1 else \
                    np.concatenate(all_embeddings, axis=0)
        train_lbl = np.zeros(len(train_emb))  # training data is all "in-distribution"

        # Also evaluate val set
        if dls.valid is not None:
            print("\nEvaluating on validation set …")
            val_embs = []
            val_lbls = []
            for ckpt in args.model_paths:
                model = build_model(ckpt, args, device)
                captured, hook = register_embedding_hook(model)
                emb, lbl = extract_embeddings(model, dls.valid, revin, captured, args, device)
                hook.remove()
                val_embs.append(emb)
                val_lbls.append(lbl)

            val_emb = val_embs[0] if len(val_embs) == 1 else val_embs[0]
            val_lbl = val_lbls[0]

            os.makedirs(args.out_dir, exist_ok=True)
            evaluate(detector, val_emb, val_lbl,
                     threshold=args.threshold,
                     out_dir=args.out_dir,
                     split_name="val",
                     model_tag=model_tag)

    # ---- EVALUATE MODE ------------------------------------------------------
    elif args.mode == "evaluate":
        print("=" * 60)
        print(f"EVALUATE MODE — scoring {args.eval_split} split")
        print("=" * 60)

        detector = MahalanobisOODDetector.load(args.load_detector)

        dl = split_map[args.eval_split]
        if dl is None:
            raise RuntimeError(f"No dataloader for split '{args.eval_split}'")

        # Extract embeddings from first model path (for scoring)
        print(f"\nExtracting embeddings from: {args.model_paths[0]}")
        model = build_model(args.model_paths[0], args, device)
        captured, hook = register_embedding_hook(model)
        emb, lbl = extract_embeddings(model, dl, revin, captured, args, device)
        hook.remove()

        os.makedirs(args.out_dir, exist_ok=True)
        evaluate(detector, emb, lbl,
                 threshold=args.threshold,
                 out_dir=args.out_dir,
                 split_name=args.eval_split,
                 model_tag=model_tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
