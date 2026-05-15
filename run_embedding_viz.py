"""
Speaker embedding visualization using t-SNE and PCA.
Uses pre-computed embeddings from exp/asv_ssl/cosine_out/emb_xvect/

Usage:
    python run_embedding_viz.py              # t-SNE 2D (default)
    python run_embedding_viz.py --method pca
    python run_embedding_viz.py --method both
    python run_embedding_viz.py --dim 3      # 3D visualization
    python run_embedding_viz.py --method both --dim 3
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

EMB_BASE = Path("exp/asv_ssl/cosine_out/emb_xvect")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("exp/embedding_viz")

DATASETS = {
    "libri_dev_enrolls":      {"level": "utt-level"},
    "libri_dev_trials_mixed": {"level": "utt-level"},
    "libri_test_enrolls":     {"level": "utt-level"},
    "libri_test_trials_mixed":{"level": "utt-level"},
}


def load_embeddings_and_speakers(dataset, level):
    base = EMB_BASE / dataset / level
    embeddings = torch.load(base / "speaker_vectors.pt").cpu().numpy()

    # idx2spk: 행번호 → 화자ID (임베딩 저장 시 함께 생성된 파일)
    idx2spk = {}
    with open(base / "idx2spk") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx2spk[int(parts[0])] = parts[1]

    assert len(idx2spk) == len(embeddings), \
        f"idx2spk({len(idx2spk)})와 embeddings({len(embeddings)}) 수가 불일치: {dataset}"

    speaker_ids = [idx2spk[i] for i in range(len(embeddings))]
    return embeddings, speaker_ids


def reduce(embeddings, method, n_components):
    if method == "tsne":
        return TSNE(n_components=n_components, random_state=42, perplexity=30).fit_transform(embeddings)
    elif method == "pca":
        return PCA(n_components=n_components, random_state=42).fit_transform(embeddings)


def get_colors(n):
    """tab20/tab20b/tab20c 조합으로 최대 60색, 초과 시 hsv로 fallback."""
    if n <= 20:
        return [plt.cm.tab20(i / 20) for i in range(n)]
    if n <= 40:
        base = [plt.cm.tab20(i / 20) for i in range(20)]
        base += [plt.cm.tab20b(i / 20) for i in range(n - 20)]
        return base
    if n <= 60:
        base = [plt.cm.tab20(i / 20) for i in range(20)]
        base += [plt.cm.tab20b(i / 20) for i in range(20)]
        base += [plt.cm.tab20c(i / 20) for i in range(n - 40)]
        return base
    return [plt.cm.hsv(i / n) for i in range(n)]


def plot_embeddings_2d(reduced, speaker_ids, title, ax, method):
    unique_spks = sorted(set(speaker_ids))
    colors = get_colors(len(unique_spks))
    spk2idx = {spk: i for i, spk in enumerate(unique_spks)}

    for spk in unique_spks:
        mask = [i for i, s in enumerate(speaker_ids) if s == spk]
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   color=colors[spk2idx[spk]], s=15, alpha=0.7, label=spk)

    ax.set_title(f"{title}\n({method.upper()}, {len(unique_spks)} spks)", fontsize=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if len(unique_spks) <= 40:
        ax.legend(fontsize=6, markerscale=1.5, ncol=2,
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)


def plot_embeddings_3d(reduced, speaker_ids, title, ax, method):
    unique_spks = sorted(set(speaker_ids))
    colors = get_colors(len(unique_spks))
    spk2idx = {spk: i for i, spk in enumerate(unique_spks)}

    for spk in unique_spks:
        mask = [i for i, s in enumerate(speaker_ids) if s == spk]
        ax.scatter(reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                   color=colors[spk2idx[spk]], s=15, alpha=0.7, label=spk)

    ax.set_title(f"{title}\n({method.upper()}, {len(unique_spks)} spks)", fontsize=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    if len(unique_spks) <= 40:
        ax.legend(fontsize=6, markerscale=1.5, ncol=2,
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)


def run(method, dim):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if dim == 3:
        fig = plt.figure(figsize=(18, 12))
        axes = [fig.add_subplot(2, 2, i+1, projection='3d') for i in range(4)]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

    for i, (dataset, info) in enumerate(DATASETS.items()):
        print(f"Processing {dataset}...")
        embeddings, speaker_ids = load_embeddings_and_speakers(dataset, info["level"])

        reduced = reduce(embeddings, method, n_components=dim)
        if dim == 3:
            plot_embeddings_3d(reduced, speaker_ids, dataset, axes[i], method)
        else:
            plot_embeddings_2d(reduced, speaker_ids, dataset, axes[i], method)

    plt.suptitle(f"Speaker Embedding Distribution ({method.upper()}, {dim}D)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"embedding_{method}_{dim}d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tsne", "pca", "both"], default="tsne")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    args = parser.parse_args()

    methods = ["tsne", "pca"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"\n--- Running {method.upper()} {args.dim}D ---")
        run(method, args.dim)


if __name__ == "__main__":
    main()
