#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["chemistry", "computer_science", "political_science", "theology"]

# New naming for reporting/plots
TYPE_ALIAS = {
    "original": "original",
    "improved": "refine",   # improve -> refine
    "new": "new",
    "rewritten": "polish",  # rewrite -> polish
}


def _extract_score(d: dict) -> Optional[float]:
    # Support both Pangram v3 and earlier results
    s = d.get("ai_likelihood")
    if isinstance(s, (int, float)):
        return float(s)
    s = d.get("fraction_ai")
    if isinstance(s, (int, float)):
        return float(s)
    return None


def _infer_type_from_dirname(dirname: str) -> Optional[str]:
    # dirname looks like "<type>_pangram_results"
    if not dirname.endswith("_pangram_results"):
        return None
    t = dirname.replace("_pangram_results", "")
    return t if t in TYPE_ALIAS else None


@dataclass
class SummaryRow:
    collection: str
    domain: str
    type: str  # original/refine/new/polish
    n: int
    mean: float
    median: float
    min: float
    max: float
    flagged_ge_0_5: int


def _median(xs: List[float]) -> float:
    xs2 = sorted(xs)
    n = len(xs2)
    mid = n // 2
    if n % 2 == 1:
        return xs2[mid]
    return (xs2[mid - 1] + xs2[mid]) / 2.0


def summarize_collection(collection: str) -> List[SummaryRow]:
    base = ROOT / "ai_improvement_results" / collection
    rows: List[SummaryRow] = []

    for dom in DOMAINS:
        dom_dir = base / dom
        if not dom_dir.exists():
            continue

        # Gather scores by old-type name (original/improved/new/rewritten)
        scores_by_type: Dict[str, List[float]] = {t: [] for t in TYPE_ALIAS.keys()}

        for subdir in dom_dir.iterdir():
            if not subdir.is_dir():
                continue
            old_type = _infer_type_from_dirname(subdir.name)
            if old_type is None:
                continue
            for p in subdir.glob("W*.json"):
                try:
                    d = json.loads(p.read_text())
                except Exception:
                    continue
                s = _extract_score(d)
                if s is not None:
                    scores_by_type[old_type].append(s)

        for old_type, xs in scores_by_type.items():
            if not xs:
                continue
            n = len(xs)
            mean = sum(xs) / n
            med = _median(xs)
            mn = min(xs)
            mx = max(xs)
            flagged = sum(1 for x in xs if x >= 0.5)
            rows.append(
                SummaryRow(
                    collection=collection,
                    domain=dom,
                    type=TYPE_ALIAS[old_type],
                    n=n,
                    mean=mean,
                    median=med,
                    min=mn,
                    max=mx,
                    flagged_ge_0_5=flagged,
                )
            )

    return rows


def write_csv(rows: List[SummaryRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "collection",
                "domain",
                "type",
                "n",
                "mean",
                "median",
                "min",
                "max",
                "flagged_ge_0_5",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.collection,
                    r.domain,
                    r.type,
                    r.n,
                    f"{r.mean:.6f}",
                    f"{r.median:.6f}",
                    f"{r.min:.6f}",
                    f"{r.max:.6f}",
                    r.flagged_ge_0_5,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize pre-AI PANGRAM results by domain and type (original/refine/new/polish)."
    )
    p.add_argument(
        "--collections",
        nargs="+",
        default=["2015_back_2013", "2025_back_2023"],
        help="Collections to summarize (default: 2015_back_2013 2025_back_2023).",
    )
    p.add_argument(
        "--out",
        default="results/preai_pangram_summary.csv",
        help="Output CSV path (default: results/preai_pangram_summary.csv).",
    )
    args = p.parse_args()

    rows: List[SummaryRow] = []
    for coll in args.collections:
        rows.extend(summarize_collection(coll))

    out_path = ROOT / args.out
    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

