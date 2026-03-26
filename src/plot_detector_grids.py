#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DOMAINS: Tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")

# Column order (left->right): original, polish, refine, new
TYPE_DIRS: Tuple[Tuple[str, str], ...] = (
    ("original", "original"),
    ("rewritten", "polish"),  # rewritten -> polish
    ("improved", "refine"),   # improved -> refine
    ("new", "new"),
)


def _collection_range_label(collection: str) -> str:
    return {
        "2015_back_2013": "Pre-LLMs 2013–2015",
        "2025_back_2023": "Post-LLMs 2023–2025",
    }.get(collection, collection)


def extract_score(detector: str, d: dict) -> Optional[float]:
    """
    Extract a comparable scalar score from a detector result JSON.

    All returned values are expected to be in [0,1].
    """
    if detector == "pangram":
        s = d.get("ai_likelihood")
        if isinstance(s, (int, float)):
            return float(s)
        s = d.get("fraction_ai")
        if isinstance(s, (int, float)):
            return float(s)
        return None

    if detector == "gptzero":
        # Observed schema: {"human": 0, "ai": 0.99, "mixed": ..., "document_class": "..."}
        s = d.get("ai")
        if isinstance(s, (int, float)):
            return float(s)
        return None

    if detector == "llm_aid":
        # Observed schema: {"ai_probability": 0.65, "model_name": "openai/gpt-5-nano", ...}
        s = d.get("ai_probability")
        if isinstance(s, (int, float)):
            return float(s)
        return None

    raise ValueError(f"Unknown detector: {detector}")


def detector_label(detector: str) -> str:
    return {
        "pangram": "Pangram",
        "gptzero": "GPTZero",
        "llm_aid": "LLM-Aid (gpt-5-nano)",
    }.get(detector, detector)


def load_scores(collection: str, detector: str) -> Dict[Tuple[str, str], List[float]]:
    """
    Load detector scores for a collection from:
      ai_improvement_results/{collection}/{domain}/{type}_{detector}_results/W*.json
    with {type} in {original, improved, new, rewritten}.

    Returns mapping (domain, plot_type_label) -> list[score].
    """
    base = ROOT / "ai_improvement_results" / collection
    out: Dict[Tuple[str, str], List[float]] = {}

    for dom in DOMAINS:
        dom_dir = base / dom
        if not dom_dir.exists():
            continue
        for type_dir, type_label in TYPE_DIRS:
            rdir = dom_dir / f"{type_dir}_{detector}_results"
            scores: List[float] = []
            if rdir.exists():
                for p in rdir.glob("W*.json"):
                    try:
                        d = json.loads(p.read_text())
                    except Exception:
                        continue
                    s = extract_score(detector, d)
                    if s is not None:
                        scores.append(s)
            out[(dom, type_label)] = scores

    return out


def plot_grid(collection: str, detector: str, output_path: Path) -> None:
    scores = load_scores(collection, detector)
    type_labels = [t[1] for t in TYPE_DIRS]

    fig, axes = plt.subplots(
        nrows=len(DOMAINS),
        ncols=len(type_labels),
        figsize=(len(type_labels) * 3.0, len(DOMAINS) * 2.2),
        sharey=True,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    for r, dom in enumerate(DOMAINS):
        for c, typ in enumerate(type_labels):
            ax = axes[r][c]
            xs = scores.get((dom, typ), [])
            if xs:
                ax.boxplot(xs, showmeans=True)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray",
                    transform=ax.transAxes,
                )

            ax.set_xticks([])  # remove the default "1"
            ax.text(
                0.5,
                -0.20,
                f"n={len(xs)}",
                ha="center",
                va="top",
                fontsize=9,
                transform=ax.transAxes,
            )
            if r == 0:
                ax.set_title(typ)
            if c == 0:
                ax.set_ylabel(dom, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    coll_label = _collection_range_label(collection)
    det_label = detector_label(detector)
    fig.suptitle(
        f"{det_label} Score Distributions ({coll_label})\n"
        "Types: Original / Polish / Refine / New",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(0.005, 0.5, f"{det_label} Score", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.06, 0.03, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 4x4 detector score grids for collections and detectors."
    )
    ap.add_argument(
        "--collections",
        nargs="+",
        default=["2015_back_2013", "2025_back_2023"],
        choices=["2015_back_2013", "2025_back_2023"],
        help="Collections to plot (default: both).",
    )
    ap.add_argument(
        "--detectors",
        nargs="+",
        default=["pangram", "gptzero", "llm_aid"],
        choices=["pangram", "gptzero", "llm_aid"],
        help="Detectors to plot (default: pangram gptzero llm_aid).",
    )
    args = ap.parse_args()

    for collection in args.collections:
        for detector in args.detectors:
            out = (
                ROOT
                / "results"
                / "figures"
                / collection
                / f"{detector}_grid.png"
            )
            plot_grid(collection, detector, out)
            print(f"Saved {detector} grid to {out}")


if __name__ == "__main__":
    main()

