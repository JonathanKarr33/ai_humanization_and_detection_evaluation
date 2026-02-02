#!/usr/bin/env python3
"""
Visualize `results/results.csv`.

This script generates ONE figure:

- `ai_likelihood_box_by_domain_faceted_by_process.png`

It’s a single figure with one subplot per process (ordered: original, rewritten, improved, new),
and each subplot shows `ai_likelihood` boxplots across domains (top N domains by row count).
Outliers (> 1.5×IQR) are shown as highlighted dots.

Usage:
  python src/visualize_results_csv.py
  python src/visualize_results_csv.py --input results/results.csv --outdir results/figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _process_order(df: pd.DataFrame) -> List[str]:
    """
    Preferred process order for plots.
    Keeps unknown/extra processes at the end (alphabetical).
    """
    if "process" not in df.columns:
        return []
    preferred = ["original", "rewritten", "improved", "new"]
    present = [p for p in preferred if (df["process"] == p).any()]
    extras = sorted([p for p in df["process"].dropna().unique().tolist() if p not in preferred])
    return present + extras


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ai_likelihood" in df.columns:
        df["ai_likelihood"] = pd.to_numeric(df["ai_likelihood"], errors="coerce")

    for col in ("domain", "process", "paper_id"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def _savefig(outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _flierprops():
    # Fliers are points outside 1.5*IQR (Tukey). Make them obvious.
    return {
        "marker": "o",
        "markersize": 3.5,
        "markerfacecolor": "crimson",
        "markeredgecolor": "crimson",
        "alpha": 0.55,
    }

def plot_ai_likelihood_by_domain_faceted_process(
    df: pd.DataFrame,
    outdir: Path,
    max_domains: int = 16,
) -> None:
    """
    One figure with 4 panels (processes): within each panel, boxplot ai_likelihood by domain.

    This is often easier to scan than separate per-process figures, while still showing domain
    variation. Outliers (beyond 1.5×IQR) are shown as highlighted dots.
    """
    if "ai_likelihood" not in df.columns or "domain" not in df.columns or "process" not in df.columns:
        return

    sns.set_theme(style="whitegrid")
    proc_order = _process_order(df)
    if not proc_order:
        return

    # Domains: keep to max_domains for readability; pick largest by row count
    dom_counts = df["domain"].value_counts()
    domains = dom_counts.head(max_domains).index.tolist()

    # Stable domain order: by overall median (desc) within selected domains
    dom_order = (
        df[df["domain"].isin(domains)]
        .groupby("domain")["ai_likelihood"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    n_panels = len(proc_order)
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * max(9, 0.55 * len(dom_order)), nrows * 5.0),
        sharey=True,
    )
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, proc in zip(axes_flat, proc_order):
        sub = df[(df["process"] == proc) & (df["domain"].isin(domains))].copy()
        sns.boxplot(
            data=sub,
            x="domain",
            y="ai_likelihood",
            order=dom_order,
            showfliers=True,
            flierprops=_flierprops(),
            ax=ax,
        )
        ax.set_title(f"{proc} (n={len(sub)})")
        ax.set_xlabel("domain")
        ax.set_ylabel("ai_likelihood")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes_flat[len(proc_order) :]:
        ax.axis("off")

    fig.suptitle(
        f"AI likelihood by domain, faceted by process (top {len(domains)} domains; fliers are > 1.5×IQR)",
        y=1.02,
        fontsize=14,
    )
    _savefig(outdir / "ai_likelihood_box_by_domain_faceted_by_process.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--max-domains",
        type=int,
        default=16,
        help="Max number of domains to facet (chosen by row count).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = load_results(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_ai_likelihood_by_domain_faceted_process(df, args.outdir, max_domains=args.max_domains)

    print(f"Wrote: {args.outdir / 'ai_likelihood_box_by_domain_faceted_by_process.png'}")


if __name__ == "__main__":
    main()

