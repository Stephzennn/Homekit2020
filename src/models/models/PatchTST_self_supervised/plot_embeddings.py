"""
plot_embeddings.py — t-SNE visualization of PatchTST embeddings.

Supports two modes:

  Finetuned model (default):
    Extracts the embedding just before the final classification linear layer
    (post-dropout, shape [B x nvars*d_model]).

  Pretrained model (--pretrain):
    The pretrain head is discarded — the embedding is the raw backbone output
    aggregated to a fixed-size vector:
        [B x nvars x d_model x num_patch]
        → last patch  [:, :, :, -1]   →  [B x nvars x d_model]
        → flatten                      →  [B x nvars*d_model]
    This is conceptually identical to what the classification head sees during
    finetuning, but read straight from the pretrained backbone.

Usage (run from the PatchTST_self_supervised directory):

  Finetuned checkpoint:
    python plot_embeddings.py \
        --checkpoint /path/to/finetuned.pth \
        --out_dir    /path/to/output \
        --c_in 8 --d_model 256 --n_layers 4 --n_heads 8 --d_ff 512 \
        --patch_len 1440 --stride 180 --context_points 10080

  Pretrained checkpoint:
    python plot_embeddings.py \
        --pretrain \
        --checkpoint /path/to/pretrained.pth \
        --out_dir    /path/to/output \
        --c_in 8 --d_model 256 --n_layers 4 --n_heads 8 --d_ff 512 \
        --patch_len 1440 --stride 180 --context_points 10080
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — mirrors what patchtst_finetune.py does
# ---------------------------------------------------------------------------
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
from sklearn.manifold import TSNE

from src.models.patchTST import PatchTST
from src.models.layers.revin import RevIN
from src.callback.patch_mask import create_patch
from datautils import get_dls


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Plot t-SNE of PatchTST pre-classification embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the fine-tuned .pth checkpoint")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory where output PNGs are saved")

    # --- Architecture (must match the checkpoint) ---
    p.add_argument("--c_in",      type=int, default=8,   help="Number of input variables (channels)")
    p.add_argument("--d_model",   type=int, default=256, help="Transformer d_model")
    p.add_argument("--n_layers",  type=int, default=6,   help="Number of Transformer layers")
    p.add_argument("--n_heads",   type=int, default=8,   help="Number of attention heads")
    p.add_argument("--d_ff",      type=int, default=512, help="Transformer FFN hidden dim")
    p.add_argument("--dropout",       type=float, default=0.2, help="Transformer dropout")
    p.add_argument("--head_dropout",  type=float, default=0.2, help="Classification head dropout")
    p.add_argument("--head_type", type=str, default="classification",
                   choices=["classification", "resnet_classification"],
                   help="Head architecture used during fine-tuning (ignored when --pretrain is set)")

    # --- Pretrain mode ---
    p.add_argument("--pretrain", action="store_true", default=False,
                   help=(
                       "Load a pretrained (masked-reconstruction) checkpoint instead of a "
                       "finetuned one. The pretrain head is discarded; the embedding is the "
                       "backbone output aggregated via last-patch + flatten."
                   ))

    # --- Patching ---
    p.add_argument("--patch_len",      type=int, default=1440, help="Patch length")
    p.add_argument("--stride",         type=int, default=180,  help="Stride between patches")
    p.add_argument("--context_points", type=int, default=10080,
                   help="Sequence length (context window)")

    # --- Data ---
    p.add_argument("--batch_size",   type=int, default=32,  help="Batch size for inference")
    p.add_argument("--num_workers",  type=int, default=0,   help="DataLoader worker processes")
    p.add_argument("--revin",        type=int, default=1,   help="Use RevIN normalization (1=yes)")

    # --- t-SNE ---
    p.add_argument("--tsne_perplexity", type=float, default=None,
                   help="t-SNE perplexity. Defaults to min(30, N//10)")
    p.add_argument("--tsne_n_iter",     type=int,   default=1000, help="t-SNE iterations")

    # --- Splits to include ---
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="Which data splits to visualize")

    return p


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(args, device):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print(f"num_patch = {num_patch}   embedding_dim = {args.c_in * args.d_model}")

    if args.pretrain:
        # Pretrain head expects (d_model, patch_len, dropout) and reconstructs patches.
        # target_dim is patch_len for the pretrain head's linear projection.
        head_type  = "pretrain"
        target_dim = args.patch_len
    else:
        head_type  = args.head_type
        target_dim = 1

    model = PatchTST(
        c_in=args.c_in,
        target_dim=target_dim,
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
        head_type=head_type,
        res_attention=False,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    mode_label = "pretrained (backbone only)" if args.pretrain else f"finetuned ({head_type} head)"
    print(f"Loaded [{mode_label}]: {args.checkpoint}")
    return model


# ---------------------------------------------------------------------------
# Forward hooks — two modes
#
# Finetuned (classification head):
#   ClassificationHead.forward:
#     x = x[:,:,:,-1]      [B x nvars x d_model]   (last patch)
#     x = flatten(x)       [B x nvars*d_model]
#     x = dropout(x)       [B x nvars*d_model]  ← hook fires HERE (post-dropout)
#     y = linear(x)        [B x 1]              ← skipped
#
# Pretrained (backbone only):
#   PatchTST.forward calls backbone → head.
#   We intercept the backbone output at the INPUT of model.head using a
#   forward_pre_hook, then aggregate it the same way the classification head
#   would: last patch → flatten → [B x nvars*d_model].
# ---------------------------------------------------------------------------
def register_finetune_embedding_hook(model):
    """Hook on head.dropout output — works for classification & resnet_classification heads."""
    captured = []

    def _hook(module, inp, out):
        captured.append(out.detach().cpu())

    handle = model.head.dropout.register_forward_hook(_hook)
    return captured, handle


def register_pretrain_embedding_hook(model):
    """
    Hook on the INPUT of model.head (= backbone output) using a pre-hook.
    Aggregates [B x nvars x d_model x num_patch] → [B x nvars*d_model]
    by taking the last patch and flattening — identical to what the
    classification head does before its linear layer.
    """
    captured = []

    def _pre_hook(module, inp):
        # inp is a tuple; inp[0] is the backbone output [B x nvars x d_model x num_patch]
        x = inp[0].detach().cpu()           # [B x nvars x d_model x num_patch]
        x = x[:, :, :, -1]                 # [B x nvars x d_model]  (last patch)
        x = x.flatten(start_dim=1)         # [B x nvars * d_model]
        captured.append(x)

    handle = model.head.register_forward_pre_hook(_pre_hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Data loading shim
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
    return _Args()


# ---------------------------------------------------------------------------
# Embedding extraction for one DataLoader
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

        # RevIN normalization
        if args.revin:
            xb = revin(xb, "norm")

        # Patching  [B x T x C] → [B x num_patch x C x patch_len]
        xb_patch, _ = create_patch(xb, args.patch_len, args.stride)

        # Forward — hook captures the embedding
        _ = model(xb_patch)

        all_emb.append(captured.pop())
        all_lbl.append(yb.reshape(-1).cpu())

    return (
        torch.cat(all_emb, dim=0).numpy(),
        torch.cat(all_lbl, dim=0).numpy(),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def run_tsne(embeddings, args, n_samples):
    perplexity = args.tsne_perplexity or min(30, max(5, n_samples // 10))
    print(f"  t-SNE: perplexity={perplexity}, n_iter={args.tsne_n_iter}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=args.tsne_n_iter,
        random_state=42,
        verbose=0,
    )
    return tsne.fit_transform(embeddings)


def save_individual_plot(z, labels, split_name, args, out_path):
    n_total = len(labels)
    n_pos   = int((labels == 1).sum())
    n_neg   = int((labels == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[labels == 0, 0], z[labels == 0, 1],
               c="#4878d0", alpha=0.55, s=18, linewidths=0,
               label=f"Flu-Negative  (n={n_neg})")
    ax.scatter(z[labels == 1, 0], z[labels == 1, 1],
               c="#d65f5f", alpha=0.75, s=28, linewidths=0,
               label=f"Flu-Positive  (n={n_pos})")

    mode_tag = "pretrained backbone" if args.pretrain else "pre-classification"
    ax.set_title(
        f"{mode_tag} embedding (t-SNE) — {split_name} set\n"
        f"patch={args.patch_len}  stride={args.stride}  "
        f"d_model={args.d_model}  |  N={n_total}",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.legend(fontsize=9, markerscale=1.8, framealpha=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_combined_plot(split_data, args, out_path):
    splits   = list(split_data.keys())
    n_splits = len(splits)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5.5))
    if n_splits == 1:
        axes = [axes]

    for ax, split_name in zip(axes, splits):
        labels, z = split_data[split_name]
        n_total = len(labels)
        n_pos   = int((labels == 1).sum())
        n_neg   = int((labels == 0).sum())

        ax.scatter(z[labels == 0, 0], z[labels == 0, 1],
                   c="#4878d0", alpha=0.55, s=14, linewidths=0,
                   label=f"Negative (n={n_neg})")
        ax.scatter(z[labels == 1, 0], z[labels == 1, 1],
                   c="#d65f5f", alpha=0.75, s=22, linewidths=0,
                   label=f"Positive (n={n_pos})")

        ax.set_title(f"{split_name}  (N={n_total})", fontsize=11)
        ax.set_xlabel("t-SNE dim 1", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", fontsize=9)
        ax.legend(fontsize=8, markerscale=1.5, framealpha=0.8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    mode_tag = "pretrained backbone" if args.pretrain else "pre-classification"
    fig.suptitle(
        f"PatchTST {mode_tag} embeddings (t-SNE)\n"
        f"patch={args.patch_len}  stride={args.stride}  "
        f"d_model={args.d_model}  embedding_dim={args.c_in * args.d_model}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model          = build_model(args, device)
    if args.pretrain:
        captured, hook = register_pretrain_embedding_hook(model)
        print("Embedding source: backbone output (last patch, flattened) — pretrain mode")
    else:
        captured, hook = register_finetune_embedding_hook(model)
        print("Embedding source: pre-classification dropout output — finetune mode")
    revin              = RevIN(num_features=args.c_in, eps=1e-5, affine=False).to(device)

    print("\nLoading data …")
    dls       = get_dls(make_data_args(args))
    split_map = {"train": dls.train, "val": dls.valid, "test": dls.test}

    split_data = {}   # {split_name: (labels, z_2d)} for combined figure

    for split_name in args.splits:
        dl = split_map[split_name]
        if dl is None:
            print(f"\n{split_name}: no data available, skipping.")
            continue

        print(f"\n--- {split_name} ---")
        embeddings, labels = extract_embeddings(model, dl, revin, captured, args, device)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"  Samples: {len(labels)}  (pos={n_pos}, neg={n_neg})")

        z = run_tsne(embeddings, args, len(labels))

        out_path = os.path.join(args.out_dir, f"embedding_tsne_{split_name}.png")
        save_individual_plot(z, labels, split_name, args, out_path)

        split_data[split_name] = (labels, z)

    hook.remove()

    if len(split_data) > 1:
        combined_path = os.path.join(args.out_dir, "embedding_tsne_all_splits.png")
        save_combined_plot(split_data, args, combined_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
