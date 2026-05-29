#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DOMAINS: Tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")

# Folder names on disk (legacy) -> label names for plots (new)
# Column order (left->right): original, polish, refine, new
TYPE_DIRS: Tuple[Tuple[str, str], ...] = (
    ("original", "original"),
    ("rewritten", "refine (abstract only)"),
    ("improved", "refine (abstract + paper)"),
    ("new", "new (article only)"),
)


def _title_type(typ: str) -> str:
    short = {
        "original": "Original",
        "refine (abstract only)": "Refine (abs. only)",
        "refine (abstract + paper)": "Refine (abs.+paper)",
        "new (article only)": "New (article only)",
    }
    return short.get(typ, typ)


def _types_suptitle_line() -> str:
    return " / ".join(_title_type(t[1]) for t in TYPE_DIRS)


def extract_pangram_score(d: dict) -> Optional[float]:
    """
    Extract a comparable scalar score from a Pangram result JSON.

    Supports both:
    - legacy: ai_likelihood (0-1)
    - newer:  fraction_ai (0-1)
    """
    s = d.get("ai_likelihood")
    if isinstance(s, (int, float)):
        return float(s)
    s = d.get("fraction_ai")
    if isinstance(s, (int, float)):
        return float(s)
    return None


def load_scores(collection: str) -> Dict[Tuple[str, str], List[float]]:
    """
    Load Pangram scores for a collection from:
      ai_improvement_results/{collection}/{domain}/{type}_pangram_results/W*.json

    Returns mapping (domain, plot_type_label) -> list[score].
    """
    base = ROOT / "ai_improvement_results" / collection
    out: Dict[Tuple[str, str], List[float]] = {}

    for dom in DOMAINS:
        dom_dir = base / dom
        if not dom_dir.exists():
            continue
        for type_dir, type_label in TYPE_DIRS:
            pdir = dom_dir / f"{type_dir}_pangram_results"
            scores: List[float] = []
            if pdir.exists():
                for p in pdir.glob("W*.json"):
                    try:
                        d = json.loads(p.read_text())
                    except Exception:
                        continue
                    s = extract_pangram_score(d)
                    if s is not None:
                        scores.append(s)
            out[(dom, type_label)] = scores

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot Pangram score distributions for 4 domains × 4 abstract types "
            "(original/refine/new/polish) in a single grid figure."
        )
    )
    ap.add_argument(
        "--collections",
        nargs="+",
        default=["2015_back_2013", "2025_back_2023"],
        choices=["2015_back_2013", "2025_back_2023"],
        help="Which collections under ai_improvement_results/ to plot (default: both).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path (only valid when plotting exactly one collection). "
            "Default: results/figures/{collection}/pangram_grid.png"
        ),
    )
    args = ap.parse_args()

    if args.output and len(args.collections) != 1:
        raise SystemExit("--output can only be used with a single --collections value")

    type_labels = [t[1] for t in TYPE_DIRS]

    for collection in args.collections:
        out_path = (
            ROOT / args.output
            if args.output
            else ROOT / "figures" / collection / "pangram_grid.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        scores = load_scores(collection)

        fig, axes = plt.subplots(
            nrows=len(DOMAINS),
            ncols=len(type_labels),
            figsize=(len(type_labels) * 3.0, len(DOMAINS) * 2.2),
            sharey=True,
        )
        fig.subplots_adjust(hspace=0.55, wspace=0.25)

        for r, dom in enumerate(DOMAINS):
            for c, typ in enumerate(type_labels):
                ax = axes[r][c] if len(DOMAINS) > 1 else axes[c]
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
                # Remove the default single-category x tick label ("1")
                ax.set_xticks([])
                n = len(xs)
                # Put sample size below each subplot
                ax.text(
                    0.5,
                    -0.20,
                    f"n={n}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    transform=ax.transAxes,
                )
                if r == 0:
                    ax.set_title(_title_type(typ), fontweight="bold")
                if c == 0:
                    ax.set_ylabel(dom, fontweight="bold")
                ax.grid(axis="y", alpha=0.3)

        title_range = {
            "2015_back_2013": "pre-LLMs 2013-2015",
            "2025_back_2023": "post-LLMs 2023-2025",
        }.get(collection, collection)

        fig.suptitle(
            f"Pangram score distributions ({title_range})\n"
            f"{_types_suptitle_line()}",
            fontsize=12,
            fontweight="bold",
        )
        # Single shared y-axis label (to avoid repeating in every subplot)
        fig.text(0.005, 0.5, "Pangram score", va="center", rotation="vertical")
        # Add extra left margin so the shared label doesn't overlap the domain labels
        fig.tight_layout(rect=[0.06, 0.03, 1, 0.93])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()

