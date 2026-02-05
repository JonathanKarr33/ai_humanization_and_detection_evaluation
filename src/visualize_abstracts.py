#!/usr/bin/env python3
"""
Visualize abstract coverage for any paper collection.

Reads:
  papers/{collection}/metadata_{collection}.jsonl

Requires each record to have:
  - domain
  - abstract
  - publication_month (preferred) or publication_date (YYYY-MM-DD)

Outputs:
  results/figures/{collection}/abstract_counts_min{MIN_WORDS}.png

Figure contents:
1) Count of abstracts >= MIN words, by domain
2) Count of abstracts >= MIN words, by month (YYYY-MM), stacked by domain
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def load_metadata_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def normalize_month(rec: dict) -> Optional[str]:
    month = rec.get("publication_month")
    if isinstance(month, str) and re.match(r"^\d{4}-\d{2}$", month):
        return month
    pub_date = rec.get("publication_date")
    if isinstance(pub_date, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", pub_date):
        return pub_date[:7]
    return None


def discover_collections(papers_dir: Path) -> list[str]:
    collections: list[str] = []
    if not papers_dir.exists():
        return collections
    for p in sorted(papers_dir.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        meta = p / f"metadata_{p.name}.jsonl"
        if meta.exists():
            collections.append(p.name)
    return collections


def render_collection(collection: str, min_words: int, outdir: Path) -> Path:
    meta_path = Path("papers") / collection / f"metadata_{collection}.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    records = load_metadata_jsonl(meta_path)
    if not records:
        raise RuntimeError(f"No records found in {meta_path}")

    filtered = []
    for r in records:
        if not r.get("domain"):
            continue
        if word_count(r.get("abstract") or "") < min_words:
            continue
        filtered.append(r)
    if not filtered:
        raise RuntimeError(f"No abstracts >= {min_words} words found in {meta_path}")

    rows = [{"domain": r.get("domain"), "month": normalize_month(r)} for r in filtered]
    df = pd.DataFrame(rows)

    domain_counts = df["domain"].value_counts().sort_values(ascending=False)

    df_month = df.dropna(subset=["month"]).copy()
    month_pivot = (
        df_month.pivot_table(index="month", columns="domain", aggfunc="size", fill_value=0)
        .sort_index()
    )

    # If we have month coverage, fill missing months with 0 to make the time axis continuous.
    if len(month_pivot) > 0:
        start = month_pivot.index.min()
        end = month_pivot.index.max()
        full = pd.period_range(start=start, end=end, freq="M").astype(str)
        month_pivot = month_pivot.reindex(full, fill_value=0)

    outdir = outdir / collection
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"abstract_counts_min{min_words}.png"

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.65)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(domain_counts.index.tolist(), domain_counts.values.tolist(), color="#4c78a8")
    ax1.set_title(f"{collection}: abstracts ≥ {min_words} words — count by domain", pad=10)
    ax1.set_xlabel("domain")
    ax1.set_ylabel("count")
    ax1.tick_params(axis="x", rotation=20)

    ax2 = fig.add_subplot(gs[1, 0])
    if len(month_pivot) == 0:
        ax2.text(
            0.5,
            0.5,
            "No publication_month/date available for month chart",
            ha="center",
            va="center",
        )
        ax2.set_axis_off()
    else:
        month_pivot.plot(kind="bar", stacked=True, ax=ax2, width=0.9)
        ax2.set_title(f"{collection}: abstracts ≥ {min_words} words — count by month (stacked)", pad=12)
        ax2.set_xlabel("month (YYYY-MM)")
        ax2.set_ylabel("count")
        if len(month_pivot.index) > 36:
            step = max(1, len(month_pivot.index) // 24)
            ticks = list(range(0, len(month_pivot.index), step))
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([str(month_pivot.index[i]) for i in ticks], rotation=45, ha="right")
        else:
            ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="domain", bbox_to_anchor=(1.01, 1.0), loc="upper left")

    fig.suptitle(f"Abstract coverage — {collection}", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.93, right=0.80)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection folder under papers/. If omitted, renders ALL collections found under papers/.",
    )
    parser.add_argument("--min-words", type=int, default=25)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    if args.collection:
        collections = [args.collection]
    else:
        collections = discover_collections(Path("papers"))
        if not collections:
            raise SystemExit("No collections found under papers/ (expected papers/{collection}/metadata_{collection}.jsonl).")

    for c in collections:
        outpath = render_collection(c, min_words=args.min_words, outdir=args.outdir)
        print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()

