#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOMAINS: Tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")

# Column/type order used everywhere else
TYPE_DIRS: Tuple[Tuple[str, str], ...] = (
    ("original", "original"),
    ("rewritten", "refine (abstract only)"),
    ("improved", "refine (abstract + paper)"),
    ("new", "new (article only)"),
)
TYPE_ORDER: Tuple[str, ...] = tuple(label for _, label in TYPE_DIRS)


def _title_type(typ: str) -> str:
    short = {
        "original": "Original",
        "refine (abstract only)": "Refine (abs. only)",
        "refine (abstract + paper)": "Refine (abs.+paper)",
        "new (article only)": "New (article only)",
    }
    return short.get(typ, typ)


def _collection_range_label(collection: str) -> str:
    return {
        "2015_back_2013": "Pre-LLMs 2013-2015",
        "2025_back_2023": "Post-LLMs 2023-2025",
    }.get(collection, collection)


def _pangram_score(d: dict) -> Optional[float]:
    s = d.get("ai_likelihood")
    if isinstance(s, (int, float)):
        return float(s)
    s = d.get("fraction_ai")
    if isinstance(s, (int, float)):
        return float(s)
    return None


def _gptzero_score(d: dict) -> Optional[float]:
    s = d.get("ai")
    if isinstance(s, (int, float)):
        return float(s)
    return None


def _rankdata(values: List[float]) -> List[float]:
    """
    Average-rank values with ties (1..n).
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearsonr(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _spearmanr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearsonr(rx, ry)


def _confusion_and_kappa(
    x: List[float],
    y: List[float],
    threshold: float,
) -> Tuple[Tuple[int, int, int, int], Optional[float]]:
    """
    Treat x (GPTZero) and y (Pangram) as AI scores in [0,1].
    Compute a symmetric 2x2 agreement table with AI = score >= threshold.

    Returns (both_ai, gptzero_only_ai, pangram_only_ai, both_not_ai) and Cohen's kappa.
    """
    both_ai = gptzero_only_ai = pangram_only_ai = both_not_ai = 0
    for gx, py in zip(x, y):
        g = gx >= threshold
        p = py >= threshold
        if g and p:
            both_ai += 1
        elif g and not p:
            gptzero_only_ai += 1
        elif not g and p:
            pangram_only_ai += 1
        else:
            both_not_ai += 1

    n = both_ai + gptzero_only_ai + pangram_only_ai + both_not_ai
    if n == 0:
        return (both_ai, gptzero_only_ai, pangram_only_ai, both_not_ai), None

    po = (both_ai + both_not_ai) / n
    p_g_pos = (both_ai + gptzero_only_ai) / n
    p_g_neg = 1 - p_g_pos
    p_p_pos = (both_ai + pangram_only_ai) / n
    p_p_neg = 1 - p_p_pos
    pe = p_g_pos * p_p_pos + p_g_neg * p_p_neg
    if pe == 1:
        return (both_ai, gptzero_only_ai, pangram_only_ai, both_not_ai), None
    kappa = (po - pe) / (1 - pe)
    return (both_ai, gptzero_only_ai, pangram_only_ai, both_not_ai), kappa


def load_pairs(collection: str) -> Tuple[List[float], List[float], List[str], List[str]]:
    """
    Load matched (gptzero_ai, pangram_ai) pairs from ai_improvement_results/{collection}/...
    across domains and types. Returns x, y, type labels, and domain labels.
    """
    base = ROOT / "ai_improvement_results" / collection
    x: List[float] = []
    y: List[float] = []
    type_labels: List[str] = []
    domain_labels: List[str] = []

    for dom in DOMAINS:
        dom_dir = base / dom
        if not dom_dir.exists():
            continue

        for type_dir, type_label in TYPE_DIRS:
            pang_dir = dom_dir / f"{type_dir}_pangram_results"
            gptz_dir = dom_dir / f"{type_dir}_gptzero_results"
            if not pang_dir.exists() or not gptz_dir.exists():
                continue

            # Match by file name (W*.json)
            for p in pang_dir.glob("W*.json"):
                g = gptz_dir / p.name
                if not g.exists():
                    continue
                try:
                    pang = json.loads(p.read_text())
                    gptz = json.loads(g.read_text())
                except Exception:
                    continue
                ps = _pangram_score(pang)
                gs = _gptzero_score(gptz)
                if ps is None or gs is None:
                    continue
                x.append(gs)
                y.append(ps)
                type_labels.append(type_label)
                domain_labels.append(dom)

    return x, y, type_labels, domain_labels


def plot_scatter(
    collection: str,
    x: List[float],
    y: List[float],
    type_labels: List[str],
    domain_labels: List[str],
    threshold: float,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _title_domain(d: str) -> str:
        return (d or "").replace("_", " ").title()

    type_order = TYPE_ORDER
    # Color by domain
    domain_palette = {
        "chemistry": "#1f77b4",
        "computer_science": "#ff7f0e",
        "political_science": "#2ca02c",
        "theology": "#d62728",
    }

    # Compact figure: reduce excess whitespace while preserving readability.
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.16, top=0.78, hspace=0.55, wspace=0.25)

    title_range = _collection_range_label(collection)
    suffix = f" - {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Pangram vs GPTZero Agreement ({title_range}){suffix}",
        fontweight="bold",
        fontsize=13,
    )

    for idx, typ in enumerate(type_order):
        r = idx // 2
        c = idx % 2
        ax = axes[r][c]
        # Filter to this type, then color-code by domain
        xs_all = [xi for xi, t in zip(x, type_labels) if t == typ]
        ys_all = [yi for yi, t in zip(y, type_labels) if t == typ]
        ds_all = [di for di, t in zip(domain_labels, type_labels) if t == typ]

        if xs_all:
            for dom in DOMAINS:
                xs = [xi for xi, d in zip(xs_all, ds_all) if d == dom]
                ys = [yi for yi, d in zip(ys_all, ds_all) if d == dom]
                if xs:
                    ax.scatter(
                        xs,
                        ys,
                        s=14,
                        alpha=0.55,
                        color=domain_palette.get(dom),
                        label=_title_domain(dom) if idx == 0 else None,  # legend only once
                    )
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
                transform=ax.transAxes,
            )

        # Diagonal + threshold lines
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.6)
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_title(_title_type(typ), fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="y", pad=6)

        pr = _pearsonr(xs_all, ys_all) if len(xs_all) == len(ys_all) else None
        sr = _spearmanr(xs_all, ys_all) if len(xs_all) == len(ys_all) else None
        (both_ai, gptzero_only_ai, pangram_only_ai, both_not_ai), kappa = _confusion_and_kappa(
            xs_all, ys_all, threshold=threshold
        )
        n = len(xs_all)
        agree = (both_ai + both_not_ai) / n if n else None

        # All stats below each panel (avoid overlaying points), compact to reduce whitespace
        line1_parts = [
            f"n={n}",
            f"r={pr:.3f}" if pr is not None else "r=NA",
            f"ρ={sr:.3f}" if sr is not None else "ρ=NA",
            f"κ@{threshold:g}={kappa:.3f}" if kappa is not None else f"κ@{threshold:g}=NA",
            f"agree@{threshold:g}={agree:.3f}" if agree is not None else f"agree@{threshold:g}=NA",
        ]
        line2_parts = [
            f"bothAI={both_ai}",
            f"bothNot={both_not_ai}",
            f"gptOnly={gptzero_only_ai}",
            f"pangOnly={pangram_only_ai}",
        ]
        below_lines = [" | ".join(line1_parts), " | ".join(line2_parts)]
        ax.text(
            0.5,
            -0.20,
            "\n".join(below_lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.supxlabel("GPTZero AI Score")
    fig.supylabel("Pangram AI Score", x=0.02)
    # Domain legend (single, from the first subplot)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Domain",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=4,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
        )

    # Subplot geometry is set explicitly above to avoid overly conservative tight_layout spacing.
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create Pangram vs GPTZero agreement scatter plots for pre-AI and post-AI collections."
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary agreement stats (default: 0.5).",
    )
    ap.add_argument(
        "--collections",
        nargs="+",
        default=["2015_back_2013", "2025_back_2023"],
        choices=["2015_back_2013", "2025_back_2023"],
        help="Collections to plot (default: both).",
    )
    args = ap.parse_args()

    for collection in args.collections:
        x, y, type_labels, domain_labels = load_pairs(collection)
        out = ROOT / "results" / "figures" / collection / "pangram_vs_gptzero.png"
        plot_scatter(
            collection,
            x,
            y,
            type_labels,
            domain_labels,
            threshold=args.threshold,
            out_path=out,
        )
        print(f"Saved {out} (n={len(x)})")


if __name__ == "__main__":
    main()

