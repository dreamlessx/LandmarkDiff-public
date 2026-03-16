#!/usr/bin/env python3
"""Generate publication-quality visualizations of HDA database statistics.

Creates figures for the MICCAI paper:
1. Procedure distribution bar chart
2. Displacement magnitude heatmap per procedure
3. Quality score distribution
4. Sample before/after pairs grid
5. Landmark displacement quiver plots

Usage:
    python scripts/visualize_hda_data.py
    python scripts/visualize_hda_data.py --output paper/figures/hda_stats.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_hda_metadata(hda_dir: Path) -> dict:
    """Load metadata from processed HDA directory."""
    meta_path = hda_dir / "metadata.json"
    with open(meta_path) as f:
        return json.load(f)


def plot_procedure_distribution(metadata: dict, ax: plt.Axes) -> None:
    """Bar chart of procedure counts."""
    proc_counts = metadata.get("procedure_counts", {})
    procs = sorted(proc_counts.keys())
    counts = [proc_counts[p] for p in procs]

    colors = {
        "rhinoplasty": "#4C72B0",
        "blepharoplasty": "#55A868",
        "rhytidectomy": "#C44E52",
        "orthognathic": "#8172B2",
    }
    bar_colors = [colors.get(p, "#999999") for p in procs]

    bars = ax.bar(range(len(procs)), counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(procs)))
    ax.set_xticklabels([p.capitalize() for p in procs], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Number of Pairs", fontsize=10)
    ax.set_title("HDA Procedure Distribution", fontsize=11, fontweight="bold")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_quality_distribution(metadata: dict, ax: plt.Axes) -> None:
    """Histogram of alignment quality scores."""
    pairs = metadata.get("pairs", {})
    qualities = [p["quality_score"] for p in pairs.values()]

    ax.hist(qualities, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axvline(
        np.median(qualities),
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {np.median(qualities):.2f}",
    )
    ax.set_xlabel("Alignment Quality Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Face Alignment Quality", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_displacement_distribution(metadata: dict, ax: plt.Axes) -> None:
    """Box plot of mean displacement by procedure."""
    pairs = metadata.get("pairs", {})

    proc_disps: dict[str, list[float]] = {}
    for p in pairs.values():
        proc = p.get("procedure", "unknown")
        disp = p.get("mean_displacement", 0)
        proc_disps.setdefault(proc, []).append(disp)

    procs = sorted(proc_disps.keys())
    data = [proc_disps[p] for p in procs]

    bp = ax.boxplot(data, labels=[p[:8].capitalize() for p in procs], patch_artist=True)

    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[: len(procs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Mean Displacement (normalized)", fontsize=10)
    ax.set_title("Surgical Displacement by Procedure", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sample_pairs(hda_dir: Path, metadata: dict, ax_grid: list, n_samples: int = 4) -> None:
    """Show sample before/after pairs."""
    pairs = metadata.get("pairs", {})
    prefixes = list(pairs.keys())

    # Select diverse samples (one per procedure)
    selected = []
    seen_procs = set()
    for prefix in prefixes:
        proc = pairs[prefix].get("procedure", "unknown")
        if proc not in seen_procs and len(selected) < n_samples:
            selected.append(prefix)
            seen_procs.add(proc)

    for i, prefix in enumerate(selected):
        if i >= len(ax_grid) // 2:
            break

        before = cv2.imread(str(hda_dir / f"{prefix}_input.png"))
        after = cv2.imread(str(hda_dir / f"{prefix}_target.png"))

        if before is not None:
            ax_grid[2 * i].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
            proc = pairs[prefix].get("procedure", "unknown")
            ax_grid[2 * i].set_title(f"Before ({proc[:5]})", fontsize=8)
        ax_grid[2 * i].axis("off")

        if after is not None:
            ax_grid[2 * i + 1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
            ax_grid[2 * i + 1].set_title("After", fontsize=8)
        ax_grid[2 * i + 1].axis("off")


def generate_figure(hda_dir: Path, output_path: Path) -> None:
    """Generate the full statistics figure."""
    if not HAS_MPL:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        sys.exit(1)

    metadata = load_hda_metadata(hda_dir)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: statistics
    ax1 = fig.add_subplot(gs[0, 0])
    plot_procedure_distribution(metadata, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_quality_distribution(metadata, ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_displacement_distribution(metadata, ax3)

    # Bottom row: sample pairs
    n_samples = 3
    sample_axes = []
    for i in range(n_samples * 2):
        ax = fig.add_subplot(gs[1, i // 2] if i < 2 else (gs[1, 1] if i < 4 else gs[1, 2]))

    # Create proper sub-axes for samples
    inner_gs = GridSpec(1, 6, figure=fig, left=0.05, right=0.95, bottom=0.05, top=0.42, wspace=0.1)
    for i in range(6):
        sample_axes.append(fig.add_subplot(inner_gs[0, i]))

    plot_sample_pairs(hda_dir, metadata, sample_axes, n_samples=3)

    # Suptitle
    total = metadata.get("total_pairs", 0)
    fig.suptitle(
        f"HDA Plastic Surgery Database -- {total} Processed Pairs\n(Rathgeb et al., CVPRW 2020)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize HDA database statistics")
    parser.add_argument("--hda-dir", type=Path, default=ROOT / "data" / "hda_processed")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "paper" / "figures" / "hda_statistics.pdf"
    )
    args = parser.parse_args()

    generate_figure(args.hda_dir, args.output)


if __name__ == "__main__":
    main()
