#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOMAINS: Tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")

# Left->right order used elsewhere
TYPE_DIRS: Tuple[Tuple[str, str], ...] = (
    ("original", "original"),
    ("rewritten", "polish"),
    ("improved", "refine"),
    ("new", "new"),
)


def _collection_range_label(collection: str) -> str:
    return {
        "2015_back_2013": "pre-AI 2013–2015",
        "2025_back_2023": "post-AI 2023–2025",
    }.get(collection, collection)


def detector_label(detector: str) -> str:
    return {
        "pangram": "Pangram",
        "gptzero": "GPTZero",
        "llm_aid": "LLM-Aid (gpt-5-nano)",
    }.get(detector, detector)


def extract_score(detector: str, d: dict) -> Optional[float]:
    if detector == "pangram":
        s = d.get("ai_likelihood")
        if isinstance(s, (int, float)):
            return float(s)
        s = d.get("fraction_ai")
        if isinstance(s, (int, float)):
            return float(s)
        return None
    if detector == "gptzero":
        s = d.get("ai")
        if isinstance(s, (int, float)):
            return float(s)
        return None
    if detector == "llm_aid":
        s = d.get("ai_probability")
        if isinstance(s, (int, float)):
            return float(s)
        return None
    raise ValueError(f"Unknown detector: {detector}")


def load_cell_scores(collection: str, detector: str, domain: str, type_dir: str) -> List[float]:
    """
    Read ai_improvement_results/{collection}/{domain}/{type_dir}_{detector}_results/W*.json
    """
    base = ROOT / "ai_improvement_results" / collection / domain / f"{type_dir}_{detector}_results"
    if not base.exists():
        return []
    xs: List[float] = []
    for p in base.glob("W*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        s = extract_score(detector, d)
        if s is not None:
            xs.append(s)
    return xs


def _median(xs: List[float]) -> float:
    xs2 = sorted(xs)
    n = len(xs2)
    mid = n // 2
    if n % 2 == 1:
        return xs2[mid]
    return (xs2[mid - 1] + xs2[mid]) / 2.0


def plot_cell(
    collection: str,
    detector: str,
    domain: str,
    type_label: str,
    scores: List[float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    det_label = detector_label(detector)
    coll_label = _collection_range_label(collection)

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    if scores:
        ax.boxplot(
            scores,
            vert=False,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "#9ecae1", "edgecolor": "#1f77b4"},
            medianprops={"color": "#2c3e50", "linewidth": 1.5},
            whiskerprops={"color": "#1f77b4"},
            capprops={"color": "#1f77b4"},
            meanprops={"marker": "D", "markerfacecolor": "#d62728", "markeredgecolor": "#d62728"},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.35, "markerfacecolor": "#1f77b4"},
        )
        mean = sum(scores) / len(scores)
        med = _median(scores)
        ax.text(0.99, 0.90, f"mean={mean:.3f}", ha="right", va="top", transform=ax.transAxes, fontsize=8)
        ax.text(0.99, 0.80, f"median={med:.3f}", ha="right", va="top", transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])

    plt.xlim(0, 1)
    plt.ylabel("")
    plt.xlabel(f"{det_label} score")
    plt.title(
        f"{det_label} distribution — {coll_label}\n{domain} / {type_label} (n={len(scores)})",
        fontweight="bold",
    )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 16 per-cell (domain × type) distribution plots per collection."
    )
    ap.add_argument(
        "--collections",
        nargs="+",
        default=["2015_back_2013", "2025_back_2023"],
        choices=["2015_back_2013", "2025_back_2023"],
        help="Collections to plot (default: both).",
    )
    ap.add_argument(
        "--detector",
        default="pangram",
        choices=["pangram", "llm_aid"],
        help="Detector to plot (default: pangram). GPTZero is intentionally excluded for cell plots.",
    )
    args = ap.parse_args()

    for collection in args.collections:
        out_dir = ROOT / "results" / "figures" / collection / f"{args.detector}_cells"
        for domain in DOMAINS:
            for type_dir, type_label in TYPE_DIRS:
                scores = load_cell_scores(collection, args.detector, domain, type_dir)
                out_path = out_dir / f"{domain}__{type_label}.png"
                plot_cell(collection, args.detector, domain, type_label, scores, out_path)
        print(f"Saved 16 plots to {out_dir}")


if __name__ == "__main__":
    main()

