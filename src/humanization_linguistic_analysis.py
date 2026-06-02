#!/usr/bin/env python3
"""
Corpus-level pre/post linguistic analysis for Undetectable humanization.

Reads humanization/{collection}/{domain}/{variant}/{paper_id}.json and compares
original_abstract vs humanized_abstract on surface features aligned with the paper
(long-token ratio, AWL density, type-token ratio, sentence length, short-sentence share).

Outputs under results/statistics/humanization_linguistics/:
  - paired_feature_rows.csv
  - tests_humanization_features.json
  - score_feature_coupling.json
  - summary.md
  - appendix_snippet.tex
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from paper_stats import _perm_paired_mean, _perm_spearman, _text_features, load_rows
from variants import VARIANT_LABEL, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
HUMANIZATION_DIR = ROOT / "humanization"
OUT_DIR = ROOT / "results" / "statistics" / "humanization_linguistics"

COLLECTIONS_DEFAULT = ("2015_back_2013", "2025_back_2023")

FEATURE_COLS = [
    "long_token_ratio",
    "awl_token_ratio",
    "type_token_ratio",
    "numeric_token_ratio",
    "nonalpha_token_ratio",
    "avg_words_per_sentence",
    "short_sentence_ratio",
    "word_count",
]

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _sentence_spans(text: str) -> List[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text or "") if p.strip()]
    return parts


def _extended_text_features(text: str) -> dict:
    feats = _text_features(text)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+\./]*", (text or "").strip())
    n = len(tokens)
    if n:
        feats["type_token_ratio"] = len({t.lower() for t in tokens}) / n
    else:
        feats["type_token_ratio"] = float("nan")

    sents = _sentence_spans(text)
    if sents:
        lengths = [len(s.split()) for s in sents]
        feats["avg_words_per_sentence"] = float(np.mean(lengths))
        feats["short_sentence_ratio"] = float(np.mean(np.array(lengths) <= 8))
        feats["sentence_count"] = float(len(sents))
    else:
        feats["avg_words_per_sentence"] = float("nan")
        feats["short_sentence_ratio"] = float("nan")
        feats["sentence_count"] = float("nan")
    return feats


def load_humanization_pairs(collections: Tuple[str, ...]) -> pd.DataFrame:
    rows: List[dict] = []
    for collection in collections:
        base = HUMANIZATION_DIR / collection
        if not base.is_dir():
            continue
        for domain_dir in sorted(base.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for variant in VARIANTS:
                var_dir = domain_dir / variant
                if not var_dir.is_dir():
                    continue
                for path in sorted(var_dir.glob("W*.json")):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    pre = payload.get("original_abstract") or ""
                    post = payload.get("humanized_abstract") or ""
                    if not pre or not post:
                        continue
                    pre_f = _extended_text_features(pre)
                    post_f = _extended_text_features(post)
                    row = {
                        "collection": collection,
                        "domain": payload.get("domain") or domain,
                        "variant_raw": variant,
                        "variant": VARIANT_LABEL.get(variant, variant),
                        "paper_id": payload.get("paper_id") or path.stem,
                    }
                    for k in FEATURE_COLS:
                        row[f"pre_{k}"] = pre_f.get(k, float("nan"))
                        row[f"post_{k}"] = post_f.get(k, float("nan"))
                        row[f"delta_{k}"] = post_f.get(k, float("nan")) - pre_f.get(k, float("nan"))
                    rows.append(row)
    return pd.DataFrame(rows)


def run_feature_permutation_tests(df: pd.DataFrame, n_perm: int, seed: int) -> List[dict]:
    out: List[dict] = []
    scopes = [("pooled", df)]
    for collection in sorted(df["collection"].unique()):
        scopes.append((collection, df[df["collection"] == collection]))

    for scope_name, sub in scopes:
        if sub.empty:
            continue
        for feat in FEATURE_COLS:
            col = f"delta_{feat}"
            diffs = sub[col].dropna().to_numpy(dtype=float)
            if diffs.size < 10:
                continue
            mean_delta, p = _perm_paired_mean(diffs, n_perm=n_perm, seed=seed + hash((scope_name, feat)) % 10000)
            pre_col = f"pre_{feat}"
            post_col = f"post_{feat}"
            out.append(
                {
                    "scope": scope_name,
                    "feature": feat,
                    "n_pairs": int(diffs.size),
                    "mean_pre": float(sub[pre_col].mean()),
                    "mean_post": float(sub[post_col].mean()),
                    "mean_delta_post_minus_pre": mean_delta,
                    "median_delta": float(np.median(diffs)),
                    "fraction_decreased": float(np.mean(diffs < 0)),
                    "p_value_permutation_two_sided": p,
                }
            )
    return out


def run_score_feature_coupling(
    df: pd.DataFrame, collections: Tuple[str, ...], n_perm: int, seed: int
) -> List[dict]:
    """Spearman correlation between feature deltas and detector score deltas (pre vs post)."""
    out: List[dict] = []
    df_pre = load_rows(collections[0], "pre") if len(collections) == 1 else pd.concat(
        [load_rows(c, "pre") for c in collections], ignore_index=True
    )
    df_post = load_rows(collections[0], "post") if len(collections) == 1 else pd.concat(
        [load_rows(c, "post") for c in collections], ignore_index=True
    )
    key = ["collection", "domain", "variant_raw", "paper_id"]
    for detector in ("pangram", "gptzero"):
        pre = df_pre[df_pre["detector"] == detector][key + ["score"]].rename(columns={"score": "score_pre"})
        post = df_post[df_post["detector"] == detector][key + ["score"]].rename(columns={"score": "score_post"})
        scores = pre.merge(post, on=key, how="inner")
        scores["delta_score"] = scores["score_post"] - scores["score_pre"]
        merged = df.merge(scores, on=key, how="inner")
        if merged.empty:
            continue
        for feat in ("long_token_ratio", "awl_token_ratio", "type_token_ratio", "avg_words_per_sentence"):
            x = merged[f"delta_{feat}"].to_numpy(dtype=float)
            y = merged["delta_score"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 20:
                continue
            rho, p = _perm_spearman(x[mask], y[mask], n_perm=min(n_perm, 2000), seed=seed + hash((detector, feat)) % 10000)
            out.append(
                {
                    "detector": detector,
                    "feature_delta": feat,
                    "n_pairs": int(mask.sum()),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                }
            )
    return out


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "---"
    return f"{x:.{nd}f}"


def write_appendix_snippet(tests: List[dict], coupling: List[dict], path: Path) -> None:
    pooled = [t for t in tests if t["scope"] == "pooled"]
    label = {
        "long_token_ratio": "Long-token ratio",
        "awl_token_ratio": "AWL token ratio",
        "type_token_ratio": "Type-token ratio",
        "numeric_token_ratio": "Numeric token ratio",
        "nonalpha_token_ratio": "Non-alphabetic token ratio",
        "avg_words_per_sentence": "Avg.\\ words per sentence",
        "short_sentence_ratio": "Short-sentence ratio ($\\leq 8$ words)",
        "word_count": "Word count",
    }
    lines = [
        "% Auto-generated by src/humanization_linguistic_analysis.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Mean pre/post linguistic features after Undetectable AI v11 humanization (pooled corpus; $n="
        f"{pooled[0]['n_pairs'] if pooled else '?'} "
        "paper-variant pairs). $\\Delta$ is post minus pre. $p$ is a two-sided permutation test on paired $\\Delta$.}",
        "\\label{tab:humanization_linguistics}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Feature & Pre & Post & $\\Delta$ & \\% decreased & $p$ \\\\",
        "\\midrule",
    ]
    for t in pooled:
        if t["feature"] not in label:
            continue
        lines.append(
            f"{label[t['feature']]} & {_fmt(t['mean_pre'])} & {_fmt(t['mean_post'])} & "
            f"{_fmt(t['mean_delta_post_minus_pre'])} & {100 * t['fraction_decreased']:.1f}\\% & "
            f"{_fmt(t['p_value_permutation_two_sided'], 4)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(tests: List[dict], coupling: List[dict], path: Path) -> None:
    lines = ["# Humanization linguistic analysis", ""]
    lines.append("## Pooled paired feature shifts (post $-$ pre)")
    lines.append("")
    lines.append("| Feature | mean pre | mean post | mean Δ | % decreased | p |")
    lines.append("|---------|----------|-----------|--------|-------------|---|")
    for t in sorted([x for x in tests if x["scope"] == "pooled"], key=lambda x: x["feature"]):
        lines.append(
            f"| {t['feature']} | {t['mean_pre']:.4f} | {t['mean_post']:.4f} | "
            f"{t['mean_delta_post_minus_pre']:+.4f} | {100*t['fraction_decreased']:.1f}% | {t['p_value_permutation_two_sided']:.4g} |"
        )
    lines.append("")
    lines.append("## Score–feature coupling (Spearman on Δfeature vs Δdetector score)")
    lines.append("")
    for c in coupling:
        lines.append(
            f"- {c['detector']} / {c['feature_delta']}: ρ={c['spearman_rho']:.3f}, "
            f"p={c['p_value']:.4g}, n={c['n_pairs']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre/post linguistic analysis of Undetectable humanization.")
    parser.add_argument(
        "--collections",
        nargs="*",
        default=list(COLLECTIONS_DEFAULT),
        help="Collection folders under humanization/",
    )
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    collections = tuple(args.collections)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_humanization_pairs(collections)
    if df.empty:
        raise SystemExit(f"No humanization pairs found under {HUMANIZATION_DIR}")

    df.to_csv(args.out_dir / "paired_feature_rows.csv", index=False)

    tests = run_feature_permutation_tests(df, n_perm=args.n_perm, seed=args.seed)
    (args.out_dir / "tests_humanization_features.json").write_text(
        json.dumps(tests, indent=2) + "\n", encoding="utf-8"
    )

    coupling = run_score_feature_coupling(df, collections, n_perm=args.n_perm, seed=args.seed)
    (args.out_dir / "score_feature_coupling.json").write_text(
        json.dumps(coupling, indent=2) + "\n", encoding="utf-8"
    )

    write_summary_md(tests, coupling, args.out_dir / "summary.md")
    write_appendix_snippet(tests, coupling, args.out_dir / "appendix_snippet.tex")

    print(f"Wrote {len(df)} paired rows to {args.out_dir}")
    print(f"  Feature tests: {len(tests)}")
    print(f"  Score couplings: {len(coupling)}")


if __name__ == "__main__":
    main()
