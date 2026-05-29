#!/usr/bin/env python3
"""
Robustness and limitation-oriented analyses for the paper.

Addresses:
  - Threshold sensitivity (FPR/FNR across tau)
  - Bootstrap confidence intervals on error rates
  - Paper-level paired shifts (original -> polish on same paper)
  - Pipeline coverage / missing-data rates
  - FPR on human-labeled text after humanization
  - Polish-induced flag rate (paired: human original vs AI polish)

Outputs per collection under results/statistics/{collection}/:
  - robustness_threshold_sensitivity.json / .csv
  - robustness_error_rates_ci.json
  - robustness_paired_polish.json
  - robustness_coverage.json
  - robustness_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from paper_stats import (
    OUT_BASE,
    PRE_BASE,
    POST_BASE,
    VARIANT_LABEL,
    load_rows,
)

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
HUMAN_VARIANT = "original"
AI_VARIANTS = ("rewritten", "improved", "new")
DETECTORS_PRIMARY = ("pangram", "gptzero")
DETECTORS_ALL = ("pangram", "gptzero", "llm_aid")
DETECTOR_LABEL = {"pangram": "Pangram", "gptzero": "GPTZero", "llm_aid": "LLM-assisted"}

FIGURES_BASE = Path(__file__).resolve().parents[1] / "results" / "figures"


def _period_label(collection: str) -> str:
    return "2013-2015" if "2013" in collection else "2023-2025"


def _flag_rate(scores: np.ndarray, tau: float) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores >= tau))


def _fnr_rate(scores: np.ndarray, tau: float) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores < tau))


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(values)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boots[i] = stat_fn(sample)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def compute_threshold_sensitivity(df_pre: pd.DataFrame, df_post: pd.DataFrame) -> List[dict]:
    rows: List[dict] = []
    for collection in df_pre["collection"].unique():
        period = _period_label(collection)
        for phase, df in [("pre", df_pre), ("post", df_post)]:
            sub = df[df["collection"] == collection]
            for det in DETECTORS_PRIMARY:
                for tau in DEFAULT_THRESHOLDS:
                    human = sub[
                        (sub["detector"] == det) & (sub["variant_raw"] == HUMAN_VARIANT)
                    ]["score"].to_numpy(dtype=float)
                    ai = sub[
                        (sub["detector"] == det) & (sub["variant_raw"].isin(AI_VARIANTS))
                    ]["score"].to_numpy(dtype=float)
                    polish = sub[
                        (sub["detector"] == det) & (sub["variant_raw"] == "rewritten")
                    ]["score"].to_numpy(dtype=float)
                    rows.append(
                        {
                            "collection": collection,
                            "period": period,
                            "phase": phase,
                            "detector": det,
                            "threshold": tau,
                            "fpr_original": _flag_rate(human, tau),
                            "fnr_ai_labeled": _fnr_rate(ai, tau),
                            "polish_flagged_rate": _flag_rate(polish, tau),
                            "n_human": int(human.size),
                            "n_ai": int(ai.size),
                            "n_polish": int(polish.size),
                        }
                    )
    return rows


def compute_error_rate_cis(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    tau: float = 0.5,
    n_boot: int = 2000,
    seed: int = 42,
) -> List[dict]:
    rows: List[dict] = []
    for collection in df_pre["collection"].unique():
        period = _period_label(collection)
        for phase, df in [("pre", df_pre), ("post", df_post)]:
            sub = df[df["collection"] == collection]
            for det in DETECTORS_PRIMARY:
                human = sub[
                    (sub["detector"] == det) & (sub["variant_raw"] == HUMAN_VARIANT)
                ]["score"].to_numpy(dtype=float)
                ai = sub[
                    (sub["detector"] == det) & (sub["variant_raw"].isin(AI_VARIANTS))
                ]["score"].to_numpy(dtype=float)

                fpr_pt, fpr_lo, fpr_hi = bootstrap_ci(
                    human, lambda x: _flag_rate(x, tau), n_boot=n_boot, seed=seed
                )
                fnr_pt, fnr_lo, fnr_hi = bootstrap_ci(
                    ai, lambda x: _fnr_rate(x, tau), n_boot=n_boot, seed=seed + 1
                )
                rows.append(
                    {
                        "collection": collection,
                        "period": period,
                        "phase": phase,
                        "detector": det,
                        "threshold": tau,
                        "fpr": fpr_pt,
                        "fpr_ci_low": fpr_lo,
                        "fpr_ci_high": fpr_hi,
                        "fnr": fnr_pt,
                        "fnr_ci_low": fnr_lo,
                        "fnr_ci_high": fnr_hi,
                        "n_human": int(human.size),
                        "n_ai": int(ai.size),
                    }
                )
    return rows


def compute_paired_polish_shifts(df_pre: pd.DataFrame, tau: float = 0.5) -> List[dict]:
    """
    For each paper, compare original vs polish scores (same detector).
  Measures how often light editing flips a below-threshold human text to flagged.
    """
    rows: List[dict] = []
    for collection in df_pre["collection"].unique():
        period = _period_label(collection)
        sub = df_pre[df_pre["collection"] == collection]
        for det in DETECTORS_PRIMARY:
            orig = sub[(sub["detector"] == det) & (sub["variant_raw"] == HUMAN_VARIANT)][
                ["paper_id", "domain", "score"]
            ].rename(columns={"score": "score_orig"})
            polish = sub[(sub["detector"] == det) & (sub["variant_raw"] == "rewritten")][
                ["paper_id", "domain", "score"]
            ].rename(columns={"score": "score_polish"})
            merged = orig.merge(polish, on=["paper_id", "domain"], how="inner")
            if merged.empty:
                continue
            delta = merged["score_polish"] - merged["score_orig"]
            orig_flag = merged["score_orig"] >= tau
            polish_flag = merged["score_polish"] >= tau
            # Human below threshold -> polish flagged (assisted-writing false positive)
            assisted_fp = (~orig_flag & polish_flag).mean()
            # Human flagged -> polish not flagged (unusual)
            flip_down = (orig_flag & ~polish_flag).mean()
            rows.append(
                {
                    "collection": collection,
                    "period": period,
                    "detector": det,
                    "threshold": tau,
                    "n_pairs": int(len(merged)),
                    "mean_score_delta_polish_minus_orig": float(delta.mean()),
                    "median_score_delta": float(delta.median()),
                    "fraction_orig_flagged": float(orig_flag.mean()),
                    "fraction_polish_flagged": float(polish_flag.mean()),
                    "assisted_fp_rate": float(assisted_fp),
                    "fraction_flag_removed": float(flip_down),
                }
            )
    return rows


def compute_coverage(collection: str) -> dict:
    """Estimate how many papers have full pre-detection vs post-humanization coverage."""
    expected_variants = ("original", "rewritten", "improved", "new")
    paper_sets: Dict[str, set] = {v: set() for v in expected_variants}
    post_paper_ids: set = set()

    pre_base = PRE_BASE / collection
    post_base = POST_BASE / collection

    for domain_dir in pre_base.iterdir() if pre_base.exists() else []:
        if not domain_dir.is_dir():
            continue
        for variant in expected_variants:
            for det in ("pangram",):
                rdir = domain_dir / f"{variant}_{det}_results"
                if rdir.exists():
                    for p in rdir.glob("W*.json"):
                        paper_sets[variant].add(p.stem)

    for domain_dir in post_base.iterdir() if post_base.exists() else []:
        if not domain_dir.is_dir():
            continue
        for variant in expected_variants:
            rdir = domain_dir / f"{variant}_pangram_results"
            if rdir.exists():
                for p in rdir.glob("W*.json"):
                    post_paper_ids.add(p.stem)

    all_pre = set.union(*paper_sets.values()) if paper_sets else set()
    complete_pre = set.intersection(*paper_sets.values()) if paper_sets else set()
    orig_only = paper_sets.get("original", set())

    post_with_pre = post_paper_ids & orig_only

    return {
        "collection": collection,
        "period": _period_label(collection),
        "n_papers_any_pre_detection": len(all_pre),
        "n_papers_all_four_variants_pre": len(complete_pre),
        "n_papers_original_pre": len(orig_only),
        "n_papers_any_post_humanized": len(post_paper_ids),
        "n_papers_post_with_original_pre": len(post_with_pre),
        "fraction_complete_pre_among_original": (
            len(complete_pre) / len(orig_only) if orig_only else float("nan")
        ),
        "fraction_post_coverage_among_original": (
            len(post_with_pre) / len(orig_only) if orig_only else float("nan")
        ),
    }


def write_summary_md(
    out_path: Path,
    threshold_rows: List[dict],
    ci_rows: List[dict],
    paired_rows: List[dict],
    coverage: List[dict],
) -> None:
    lines = ["# Robustness Analysis Summary", ""]

    lines.append("## Threshold sensitivity (2023--2025, pre-humanization, tau)")
    lines.append("")
    sub = [
        r
        for r in threshold_rows
        if r["period"] == "2023--2025" and r["phase"] == "pre" and r["threshold"] in (0.4, 0.5, 0.6)
    ]
    for r in sorted(sub, key=lambda x: (x["detector"], x["threshold"])):
        lines.append(
            f"- {r['detector']} tau={r['threshold']:.1f}: "
            f"FPR={r['fpr_original']*100:.1f}%, FNR={r['fnr_ai_labeled']*100:.1f}%, "
            f"polish flagged={r['polish_flagged_rate']*100:.1f}%"
        )

    lines.append("")
    lines.append("## Bootstrap 95% CI at tau=0.5 (2023--2025, pre)")
    for r in ci_rows:
        if r["period"] != "2023--2025" or r["phase"] != "pre":
            continue
        lines.append(
            f"- {r['detector']}: FPR {r['fpr']*100:.1f}% "
            f"[{r['fpr_ci_low']*100:.1f}, {r['fpr_ci_high']*100:.1f}], "
            f"FNR {r['fnr']*100:.1f}% [{r['fnr_ci_low']*100:.1f}, {r['fnr_ci_high']*100:.1f}]"
        )

    lines.append("")
    lines.append("## Paired original->polish (2023--2025, pre, tau=0.5)")
    for r in paired_rows:
        if r["period"] != "2023--2025":
            continue
        lines.append(
            f"- {r['detector']}: n={r['n_pairs']}, mean Δscore={r['mean_score_delta_polish_minus_orig']:.3f}, "
            f"assisted-FP rate (human clear, polish flagged)={r['assisted_fp_rate']*100:.1f}%"
        )

    lines.append("")
    lines.append("## Pipeline coverage")
    for c in coverage:
        lines.append(
            f"- {c['period']}: {c['n_papers_all_four_variants_pre']} papers with all 4 variants pre-detection "
            f"({c['fraction_complete_pre_among_original']*100:.1f}% of originals); "
            f"post-humanization coverage {c['fraction_post_coverage_among_original']*100:.1f}% of originals"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_error_rate_table(
    df_pre: pd.DataFrame, df_post: pd.DataFrame, tau: float = 0.5
) -> List[dict]:
    rows: List[dict] = []
    for collection in df_pre["collection"].unique():
        period = _period_label(collection)
        for phase, df in [("pre", df_pre), ("post", df_post)]:
            sub = df[df["collection"] == collection]
            for det in DETECTORS_ALL:
                human = sub[
                    (sub["detector"] == det) & (sub["variant_raw"] == HUMAN_VARIANT)
                ]["score"].to_numpy(dtype=float)
                ai = sub[
                    (sub["detector"] == det) & (sub["variant_raw"].isin(AI_VARIANTS))
                ]["score"].to_numpy(dtype=float)
                rows.append(
                    {
                        "collection": collection,
                        "period": period,
                        "phase": phase,
                        "detector": det,
                        "detector_label": DETECTOR_LABEL.get(det, det),
                        "threshold": tau,
                        "fpr": _flag_rate(human, tau),
                        "fnr": _fnr_rate(ai, tau),
                        "n_human": int(human.size),
                        "n_ai": int(ai.size),
                    }
                )
    return rows


def _pr_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Precision-recall curve over unique score thresholds; returns P, R, AUC-PR."""
    if y_true.size == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-scores)
    y = y_true[order].astype(int)
    s = scores[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    n_pos = max(int(y.sum()), 1)
    recall = tp / n_pos
    # Add endpoints
    prec = np.concatenate([[1.0], precision, [precision[-1] if precision.size else 0.0]])
    rec = np.concatenate([[0.0], recall, [1.0]])
    auc = float(np.trapezoid(prec, rec))
    return prec, rec, auc


def compute_pr_metrics_for_collection(df_pre: pd.DataFrame, collection: str) -> List[dict]:
    rows: List[dict] = []
    period = _period_label(collection)
    block = df_pre[df_pre["collection"] == collection]
    for det in DETECTORS_ALL:
        d = block[block["detector"] == det]
        if d.empty:
            continue
        y = np.where(d["variant_raw"].to_numpy() == HUMAN_VARIANT, 0, 1)
        scores = d["score"].to_numpy(dtype=float)
        prec, rec, auc = _pr_curve(y, scores)
        row = {
            "collection": collection,
            "period": period,
            "detector": det,
            "auc_pr": auc,
            "n": int(len(d)),
            "n_human": int((y == 0).sum()),
            "n_ai": int((y == 1).sum()),
        }
        if prec.size > 2:
            idx = np.linspace(0, len(rec) - 1, min(50, len(rec))).astype(int)
            row["recall_curve"] = [float(rec[i]) for i in idx]
            row["precision_curve"] = [float(prec[i]) for i in idx]
        rows.append(row)
    return rows


def plot_pr_curves(pr_rows: List[dict], collection: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    period_rows = [r for r in pr_rows if r["collection"] == collection]
    if not period_rows:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"pangram": "#2563eb", "gptzero": "#dc2626", "llm_aid": "#16a34a"}
    for r in period_rows:
        if "recall_curve" not in r:
            continue
        ax.plot(
            r["recall_curve"],
            r["precision_curve"],
            label=f"{DETECTOR_LABEL.get(r['detector'], r['detector'])} (AUC={r['auc_pr']:.2f})",
            color=colors.get(r["detector"], "gray"),
            linewidth=2,
        )
    ax.set_xlabel("Recall (AI-labeled capture rate)")
    ax.set_ylabel("Precision (flagged items that are AI-labeled)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Precision-recall ({_period_label(collection)}, pre-humanization)")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residuals of y ~ a + b*x."""
    if x.size < 3 or np.std(x) == 0:
        return y - np.mean(y)
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_length_partial_correlations(df_pre: pd.DataFrame, collection: str) -> List[dict]:
    from paper_stats import _text_features

    rows: List[dict] = []
    period = _period_label(collection)
    text_base = (
        df_pre[df_pre["collection"] == collection]
        .query("detector == 'pangram'")
        [["domain", "variant_raw", "paper_id", "text"]]
        .drop_duplicates(subset=["domain", "variant_raw", "paper_id"])
    )
    if text_base.empty:
        return rows
    feats = text_base["text"].map(_text_features).apply(pd.Series)
    text_df = pd.concat([text_base.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    feature_cols = [
        "numeric_token_ratio",
        "nonalpha_token_ratio",
        "long_token_ratio",
        "acronym_ratio",
        "awl_token_ratio",
    ]
    key = ["domain", "variant_raw", "paper_id"]

    for det in DETECTORS_PRIMARY:
        det_df = df_pre[(df_pre["collection"] == collection) & (df_pre["detector"] == det)]
        merged = det_df[key + ["score"]].merge(text_df[key + feature_cols + ["word_count"]], on=key)
        for feat in feature_cols:
            sub = merged[["domain", "score", feat, "word_count"]].dropna()
            if len(sub) < 30:
                continue
            log_wc = np.log1p(sub["word_count"].to_numpy(dtype=float))
            y = sub["score"].to_numpy(dtype=float)
            x = sub[feat].to_numpy(dtype=float)
            d = sub["domain"].to_numpy()

            rho_raw = _spearman_np(x, y)
            y_res = _residualize(y, log_wc)
            x_res = _residualize(x, log_wc)
            rho_len = _spearman_np(x_res, y_res)

            # domain center then length partial
            y_dc = y - np.array([y[d == dom].mean() for dom in d])
            x_dc = x - np.array([x[d == dom].mean() for dom in d])
            y_dc_res = _residualize(y_dc, log_wc)
            x_dc_res = _residualize(x_dc, log_wc)
            rho_len_domain = _spearman_np(x_dc_res, y_dc_res)

            rows.append(
                {
                    "collection": collection,
                    "period": period,
                    "detector": det,
                    "feature": feat,
                    "n": int(len(sub)),
                    "spearman_rho": rho_raw,
                    "spearman_rho_partial_log_word_count": rho_len,
                    "spearman_rho_partial_log_wc_domain_centered": rho_len_domain,
                }
            )
    return rows


def run_collection(collection: str, tau: float, n_boot: int, seed: int) -> None:
    out_dir = OUT_BASE / collection
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pre = load_rows(collection, "pre")
    df_post = load_rows(collection, "post")
    if df_pre.empty:
        raise FileNotFoundError(f"No pre rows for {collection}")

    threshold_rows = compute_threshold_sensitivity(df_pre, df_post)
    ci_rows = compute_error_rate_cis(df_pre, df_post, tau=tau, n_boot=n_boot, seed=seed)
    paired_rows = compute_paired_polish_shifts(df_pre, tau=tau)
    coverage = compute_coverage(collection)

    (out_dir / "robustness_threshold_sensitivity.json").write_text(
        json.dumps(threshold_rows, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(threshold_rows).to_csv(out_dir / "robustness_threshold_sensitivity.csv", index=False)

    (out_dir / "robustness_error_rates_ci.json").write_text(
        json.dumps(ci_rows, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "robustness_paired_polish.json").write_text(
        json.dumps(paired_rows, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "robustness_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )

    error_table = compute_error_rate_table(df_pre, df_post, tau=tau)
    (out_dir / "error_rates_all_detectors.json").write_text(
        json.dumps(error_table, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(error_table).to_csv(out_dir / "error_rates_all_detectors.csv", index=False)

    pr_rows = compute_pr_metrics_for_collection(df_pre, collection)

    (out_dir / "pr_curve_metrics.json").write_text(
        json.dumps(pr_rows, indent=2) + "\n", encoding="utf-8"
    )
    fig_path = FIGURES_BASE / collection / "pr_curves_pre.png"
    plot_pr_curves(pr_rows, collection, fig_path)

    length_partial = compute_length_partial_correlations(df_pre, collection)
    (out_dir / "length_partial_correlations.json").write_text(
        json.dumps(length_partial, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(length_partial).to_csv(out_dir / "length_partial_correlations.csv", index=False)

    write_summary_md(
        out_dir / "robustness_summary.md",
        threshold_rows,
        ci_rows,
        paired_rows,
        [coverage],
    )
    # Append PR and length partial to summary
    with (out_dir / "robustness_summary.md").open("a", encoding="utf-8") as f:
        f.write("\n## Precision--recall (pre-humanization)\n")
        for r in pr_rows:
            f.write(f"- {r['detector']}: AUC-PR={r['auc_pr']:.3f} (n={r['n']})\n")
        f.write("\n## Length-partial Spearman (2023--2025, log word count)\n")
        for r in length_partial:
            if r["period"] != _period_label(collection):
                continue
            if r["feature"] in ("long_token_ratio", "awl_token_ratio"):
                f.write(
                    f"- {r['detector']} {r['feature']}: raw={r['spearman_rho']:.3f}, "
                    f"partial={r['spearman_rho_partial_log_word_count']:.3f}\n"
                )

    print(f"Robustness: {collection}")
    print(f"  Threshold grid: {len(threshold_rows)} rows")
    print(f"  Paired polish: {len(paired_rows)} detector summaries")
    print(f"  Coverage: {coverage['fraction_complete_pre_among_original']:.1%} complete pre pipeline")
    print(f"  PR figure: {fig_path}")
    if pr_rows:
        print(f"  AUC-PR: " + ", ".join(f"{r['detector']}={r['auc_pr']:.3f}" for r in pr_rows))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness analyses for paper limitations.")
    p.add_argument(
        "--collections",
        nargs="*",
        default=["2015_back_2013", "2025_back_2023"],
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for c in args.collections:
        run_collection(c, tau=args.threshold, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
