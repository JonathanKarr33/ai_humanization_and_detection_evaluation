#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _extract_pangram_score(d: dict) -> Optional[float]:
    """
    Extract a comparable scalar score from a Pangram result.

    - Prefer legacy 'ai_likelihood' if present (0-1).
    - Otherwise fall back to 'fraction_ai' if present (0-1).
    """
    s = d.get("ai_likelihood")
    if isinstance(s, (int, float)):
        return float(s)
    s = d.get("fraction_ai")
    if isinstance(s, (int, float)):
        return float(s)
    return None


def load_scores_from_dir(base: Path) -> Dict[str, List[float]]:
    """
    Load ai_likelihood scores from a directory tree of Pangram per-paper JSONs.

    Returns a mapping variant -> list[score], where variant name is inferred from
    directory names like '{variant}_pangram_results'.
    """
    scores: Dict[str, List[float]] = {}
    if not base.exists():
        return scores

    for var_dir in base.glob("*_pangram_results"):
        if not var_dir.is_dir():
            continue
        var = var_dir.name.replace("_pangram_results", "")
        vscores: List[float] = []
        for p in var_dir.glob("W*.json"):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            s = _extract_pangram_score(data)
            if s is not None:
                vscores.append(s)
        if vscores:
            scores[var] = vscores
    return scores


def load_original_pangram_scores(domain_filter: str | None = None) -> List[float]:
    path = ROOT / "pangram_abstracts_results.json"
    if not path.exists():
        return []
    scores: List[float] = []
    with path.open() as f:
        data = json.load(f)
    for d in data:
        if domain_filter and d.get("domain") != domain_filter:
            continue
        s = _extract_pangram_score(d)
        if s is not None:
            scores.append(s)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Pangram ai_likelihood distributions for original/improved/new texts "
            "vs their humanized equivalents, and output a single chart."
        )
    )
    parser.add_argument(
        "--collection",
        default="2025_back_2023",
        help="Collection (default: 2025_back_2023).",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain filter (e.g., 'chemistry'); if omitted, use all domains.",
    )
    parser.add_argument(
        "--output",
        default="results/pangram_distributions.png",
        help="Path to save the figure (default: results/pangram_distributions.png).",
    )
    args = parser.parse_args()

    collection = args.collection
    domain = args.domain

    # 1) Original Pangram scores (raw texts) for original abstracts
    orig_raw_scores = load_original_pangram_scores(domain_filter=domain)

    # 2) Original/improved/new raw Pangram scores (if available) from ai_improvement_results
    ai_imp_base = ROOT / "ai_improvement_results" / collection
    raw_variant_scores: Dict[str, List[float]] = {}
    if ai_imp_base.exists():
        domains: Tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")
        for dom in domains:
            if domain and dom != domain:
                continue
            dom_dir = ai_imp_base / dom
            if not dom_dir.exists():
                continue
            dom_scores = load_scores_from_dir(dom_dir)
            for var, vals in dom_scores.items():
                raw_variant_scores.setdefault(var, []).extend(vals)

    # 3) Humanized Pangram scores from humanization_results
    human_base = ROOT / "humanization_results" / collection
    human_variant_scores: Dict[str, List[float]] = {}
    if human_base.exists():
        domains: Tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")
        for dom in domains:
            if domain and dom != domain:
                continue
            dom_dir = human_base / dom
            if not dom_dir.exists():
                continue
            dom_scores = load_scores_from_dir(dom_dir)
            for var, vals in dom_scores.items():
                human_variant_scores.setdefault(var, []).extend(vals)

    # Prepare 6 series: original_raw, original_humanized, improved_raw, improved_humanized, new_raw, new_humanized
    series_map = {
        "original_raw": orig_raw_scores,
        "original_humanized": human_variant_scores.get("original", []),
        "improved_raw": raw_variant_scores.get("improved", []),
        "improved_humanized": human_variant_scores.get("improved", []),
        "new_raw": raw_variant_scores.get("new", []),
        "new_humanized": human_variant_scores.get("new", []),
    }

    if not any(series_map.values()):
        print("No Pangram scores found to plot. Make sure Pangram scripts have been run.")
        return

    # 2x3 grid: row 0 = raw, row 1 = humanized; columns = original, improved, new
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    layout = [
        ("original_raw", 0, 0),
        ("improved_raw", 0, 1),
        ("new_raw", 0, 2),
        ("original_humanized", 1, 0),
        ("improved_humanized", 1, 1),
        ("new_humanized", 1, 2),
    ]

    for label, r, c in layout:
        ax = axes[r][c]
        vals = series_map[label]
        if vals:
            ax.boxplot(vals, showmeans=True)
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
        ax.set_title(label.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
        if c == 0:
            ax.set_ylabel("PANGRAM ai_likelihood")

    if domain:
        fig.suptitle(f"PANGRAM scores (domain={domain})\noriginal/improved/new vs humanized", fontsize=12)
    else:
        fig.suptitle("PANGRAM scores (all domains)\noriginal/improved/new vs humanized", fontsize=12)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    print(f"Saved Pangram distribution figure to {out_path}")


if __name__ == "__main__":
    main()

