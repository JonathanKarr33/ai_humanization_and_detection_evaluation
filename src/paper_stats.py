#!/usr/bin/env python3
"""
Run statistical analyses for paper-ready numeric results and auto-pick one example case.

Outputs are written to:
  results/statistics/{collection}/
    - rows_pre.csv
    - rows_post.csv
    - tests_humanization.json
    - tests_domain_stem_vs_nonstem.json
    - tests_variant_effect.json
    - tests_detector_agreement.json
    - tests_text_features.json
    - summary_for_paper.md
    - paper_insert.txt
    - example_case.json
    - example_snippet.tex
    - pangram_score_summaries_pooled.csv
    - pangram_score_summaries_by_domain.csv

``awl_token_ratio`` uses Coxhead's Academic Word List (AWL) headwords in
``data/awl_headwords.txt`` with Porter stemming (NLTK) and a few US spelling variants.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parents[1]

# Coxhead Academic Word List headwords (Victoria University of Wellington PDF).
# US spellings are added for stems that differ under Porter stemming (e.g. analyse/analyze).
AWL_HEADWORDS_PATH = ROOT / "data" / "awl_headwords.txt"
_AWL_US_SPELLING = {
    "analyse": "analyze",
    "labour": "labor",
    "licence": "license",
    "utilise": "utilize",
}

PRE_BASE = ROOT / "ai_improvement_results"
POST_BASE = ROOT / "humanization_results"
OUT_BASE = ROOT / "results" / "statistics"

DOMAINS = ("chemistry", "computer_science", "political_science", "theology")
STEM = {"chemistry", "computer_science"}
NON_STEM = {"political_science", "theology"}
VARIANTS = ("original", "rewritten", "improved", "new")
VARIANT_LABEL = {
    "original": "original",
    "rewritten": "refine (abstract only)",
    "improved": "refine (abstract + paper)",
    "new": "new (article only)",
}
DETECTORS = ("pangram", "gptzero", "llm_aid", "llm_assisted")


@dataclass
class TestResult:
    n: int
    statistic: float
    p_value: float
    notes: str = ""


def count_outliers_iqr(values: List[float]) -> int:
    """Boxplot-style fliers: outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (median-based quartiles)."""
    if len(values) < 4:
        return 0
    xs = sorted(values)

    def median(arr: List[float]) -> float:
        n = len(arr)
        mid = n // 2
        if n % 2 == 1:
            return arr[mid]
        return (arr[mid - 1] + arr[mid]) / 2.0

    n = len(xs)
    if n % 2 == 0:
        lower, upper = xs[: n // 2], xs[n // 2 :]
    else:
        lower, upper = xs[: n // 2], xs[n // 2 + 1 :]
    if not lower or not upper:
        return 0
    q1, q3 = median(lower), median(upper)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return sum(1 for v in xs if v < low or v > high)


def pangram_score_summaries(df: pd.DataFrame, *, by_domain: bool) -> pd.DataFrame:
    """Mean, SD, IQR, n, and 1.5*IQR outlier counts for Pangram scores."""
    sub = df[df["detector"] == "pangram"].copy()
    if sub.empty:
        return pd.DataFrame()

    group_cols = ["collection", "phase", "variant_raw"]
    if by_domain:
        group_cols.insert(2, "domain")

    rows: List[dict] = []
    for keys, grp in sub.groupby(group_cols, sort=True):
        if by_domain:
            collection, phase, domain, variant = keys
        else:
            collection, phase, variant = keys
            domain = None
        scores = grp["score"].to_numpy(dtype=float)
        if scores.size == 0:
            continue
        rows.append(
            {
                "collection": collection,
                "phase": phase,
                "domain": domain,
                "variant_raw": variant,
                "variant": VARIANT_LABEL.get(variant, variant),
                "n": int(scores.size),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
                "p25": float(np.percentile(scores, 25)),
                "p75": float(np.percentile(scores, 75)),
                "outliers_iqr": count_outliers_iqr(scores.tolist()),
            }
        )
    return pd.DataFrame(rows)


def export_pangram_summaries(out_dir: Path, df_pre: pd.DataFrame, df_post: pd.DataFrame) -> None:
    pooled = pd.concat(
        [
            pangram_score_summaries(df_pre, by_domain=False),
            pangram_score_summaries(df_post, by_domain=False),
        ],
        ignore_index=True,
    )
    by_domain = pd.concat(
        [
            pangram_score_summaries(df_pre, by_domain=True),
            pangram_score_summaries(df_post, by_domain=True),
        ],
        ignore_index=True,
    )
    pooled.to_csv(out_dir / "pangram_score_summaries_pooled.csv", index=False)
    by_domain.to_csv(out_dir / "pangram_score_summaries_by_domain.csv", index=False)


def _safe_float(v: object) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def score_for_detector(detector: str, payload: dict) -> Optional[float]:
    if detector == "pangram":
        return _safe_float(payload.get("ai_likelihood")) or _safe_float(payload.get("fraction_ai"))
    if detector == "gptzero":
        return _safe_float(payload.get("ai"))
    if detector in {"llm_aid", "llm_assisted"}:
        return _safe_float(payload.get("ai_probability")) or _safe_float(payload.get("ai_likelihood"))
    return None


def load_rows(collection: str, phase: str) -> pd.DataFrame:
    assert phase in {"pre", "post"}
    base = (PRE_BASE if phase == "pre" else POST_BASE) / collection
    rows: List[dict] = []
    if not base.exists():
        return pd.DataFrame()

    for domain in DOMAINS:
        dom_dir = base / domain
        if not dom_dir.exists():
            continue
        for variant in VARIANTS:
            for detector in DETECTORS:
                result_dir = dom_dir / f"{variant}_{detector}_results"
                if not result_dir.exists():
                    continue
                for p in sorted(result_dir.glob("W*.json")):
                    try:
                        payload = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    score = score_for_detector(detector, payload)
                    if score is None:
                        continue
                    paper_id = payload.get("paper_id") or p.stem
                    text = payload.get("text")
                    rows.append(
                        {
                            "collection": collection,
                            "phase": phase,
                            "domain": payload.get("domain") or domain,
                            "domain_group": "stem" if domain in STEM else "non_stem",
                            "variant_raw": variant,
                            "variant": VARIANT_LABEL.get(variant, variant),
                            "detector": detector,
                            "paper_id": paper_id,
                            "score": float(score),
                            "path": str(p.relative_to(ROOT)),
                            "text": text if isinstance(text, str) else "",
                        }
                    )
    return pd.DataFrame(rows)


def _perm_paired_mean(diffs: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diffs.size))
    perm = np.mean(signs * diffs, axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1.0) / (n_perm + 1.0)
    return obs, float(p)


def _perm_independent_mean(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = a.size
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        stat = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(stat) >= abs(obs):
            count += 1
    p = (count + 1.0) / (n_perm + 1.0)
    return obs, float(p)


def _anova_f(groups: List[np.ndarray]) -> float:
    nonempty = [g for g in groups if g.size > 0]
    if len(nonempty) < 2:
        return float("nan")
    all_vals = np.concatenate(nonempty)
    grand = float(np.mean(all_vals))
    ss_between = 0.0
    ss_within = 0.0
    for g in nonempty:
        m = float(np.mean(g))
        ss_between += g.size * (m - grand) ** 2
        ss_within += float(np.sum((g - m) ** 2))
    df_between = len(nonempty) - 1
    df_within = all_vals.size - len(nonempty)
    if df_within <= 0 or ss_within == 0.0:
        return float("nan")
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return float(ms_between / ms_within)


def _perm_anova(groups: Dict[str, np.ndarray], n_perm: int, seed: int) -> Tuple[float, float]:
    labels = []
    vals = []
    for k, arr in groups.items():
        for x in arr:
            labels.append(k)
            vals.append(float(x))
    if len(vals) < 3:
        return float("nan"), float("nan")
    vals_arr = np.array(vals, dtype=float)
    labels_arr = np.array(labels)
    keys = sorted(groups.keys())
    obs = _anova_f([vals_arr[labels_arr == k] for k in keys])
    if math.isnan(obs):
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels_arr)
        stat = _anova_f([vals_arr[perm_labels == k] for k in keys])
        if not math.isnan(stat) and stat >= obs:
            count += 1
    p = (count + 1.0) / (n_perm + 1.0)
    return float(obs), float(p)


def _bootstrap_ci_mean(vals: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    if vals.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = np.mean(vals[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _rank_array(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = _rank_array(x)
    ry = _rank_array(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    # Binary labels expected: 0/1
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    po = float(np.mean(a == b))
    p_a1 = float(np.mean(a == 1))
    p_b1 = float(np.mean(b == 1))
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_e >= 1.0:
        return float("nan")
    return float((po - p_e) / (1.0 - p_e))


def run_humanization_tests(df_pre: pd.DataFrame, df_post: pd.DataFrame, n_perm: int, seed: int) -> List[dict]:
    out: List[dict] = []
    if df_pre.empty or df_post.empty:
        return out
    key_cols = ["collection", "domain", "variant_raw", "paper_id", "detector"]
    merged = (
        df_pre[key_cols + ["score"]]
        .rename(columns={"score": "score_pre"})
        .merge(
            df_post[key_cols + ["score"]].rename(columns={"score": "score_post"}),
            on=key_cols,
            how="inner",
        )
    )
    if merged.empty:
        return out
    for detector in sorted(merged["detector"].unique()):
        sub = merged[merged["detector"] == detector].copy()
        diffs = (sub["score_post"] - sub["score_pre"]).to_numpy(dtype=float)
        if diffs.size < 5:
            continue
        stat, p = _perm_paired_mean(diffs, n_perm=n_perm, seed=seed + hash(detector) % 10000)
        ci_lo, ci_hi = _bootstrap_ci_mean(diffs, n_boot=3000, seed=seed + 100)
        out.append(
            {
                "detector": detector,
                "n_pairs": int(diffs.size),
                "mean_pre": float(np.mean(sub["score_pre"])),
                "mean_post": float(np.mean(sub["score_post"])),
                "mean_diff_post_minus_pre": stat,
                "mean_diff_95ci": [ci_lo, ci_hi],
                "p_value_permutation_two_sided": p,
            }
        )
    return out


def run_domain_tests(df: pd.DataFrame, n_perm: int, seed: int) -> List[dict]:
    out: List[dict] = []
    if df.empty:
        return out
    for phase in sorted(df["phase"].unique()):
        for detector in sorted(df["detector"].unique()):
            sub = df[(df["phase"] == phase) & (df["detector"] == detector)]
            if sub.empty:
                continue
            stem = sub[sub["domain"].isin(STEM)]["score"].to_numpy(dtype=float)
            non = sub[sub["domain"].isin(NON_STEM)]["score"].to_numpy(dtype=float)
            if stem.size < 5 or non.size < 5:
                continue
            stat, p = _perm_independent_mean(non, stem, n_perm=n_perm, seed=seed + len(out) + 17)
            out.append(
                {
                    "phase": phase,
                    "detector": detector,
                    "n_non_stem": int(non.size),
                    "n_stem": int(stem.size),
                    "mean_non_stem_minus_stem": stat,
                    "mean_non_stem": float(np.mean(non)),
                    "mean_stem": float(np.mean(stem)),
                    "p_value_permutation_two_sided": p,
                }
            )
    return out


def run_variant_effect_tests(df: pd.DataFrame, n_perm: int, seed: int) -> List[dict]:
    out: List[dict] = []
    if df.empty:
        return out
    for phase in sorted(df["phase"].unique()):
        for detector in sorted(df["detector"].unique()):
            sub = df[(df["phase"] == phase) & (df["detector"] == detector)]
            if sub.empty:
                continue
            groups = {
                v: sub[sub["variant_raw"] == v]["score"].to_numpy(dtype=float)
                for v in VARIANTS
            }
            nonempty = sum(1 for arr in groups.values() if arr.size > 0)
            if nonempty < 2:
                continue
            f_stat, p = _perm_anova(groups, n_perm=n_perm, seed=seed + len(out) + 99)
            if math.isnan(f_stat):
                continue
            out.append(
                {
                    "phase": phase,
                    "detector": detector,
                    "n_total": int(sum(arr.size for arr in groups.values())),
                    "group_sizes": {k: int(v.size) for k, v in groups.items()},
                    "f_statistic": f_stat,
                    "p_value_permutation": p,
                }
            )
    return out


def run_agreement_tests(df: pd.DataFrame, threshold: float) -> List[dict]:
    out: List[dict] = []
    if df.empty:
        return out
    key_cols = ["collection", "phase", "domain", "variant_raw", "paper_id"]
    pang = df[df["detector"] == "pangram"][key_cols + ["score"]].rename(columns={"score": "pangram"})
    gptz = df[df["detector"] == "gptzero"][key_cols + ["score"]].rename(columns={"score": "gptzero"})
    pair = pang.merge(gptz, on=key_cols, how="inner")
    if pair.empty:
        return out
    for phase in sorted(pair["phase"].unique()):
        sub = pair[pair["phase"] == phase]
        x = sub["gptzero"].to_numpy(dtype=float)
        y = sub["pangram"].to_numpy(dtype=float)
        rho = _spearman(x, y)
        bx = (x >= threshold).astype(int)
        by = (y >= threshold).astype(int)
        kappa = _cohen_kappa(bx, by)
        out.append(
            {
                "phase": phase,
                "n_pairs": int(sub.shape[0]),
                "spearman_rho": rho,
                "cohen_kappa_thresholded": kappa,
                "threshold": threshold,
                "positive_rate_gptzero": float(np.mean(bx)),
                "positive_rate_pangram": float(np.mean(by)),
            }
        )
    return out


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\+\./]*")


@functools.lru_cache(maxsize=1)
def _awl_stemmer_and_stems() -> Tuple[PorterStemmer, frozenset]:
    """Porter stems of AWL headwords plus common US spelling variants."""
    stemmer = PorterStemmer()
    if not AWL_HEADWORDS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {AWL_HEADWORDS_PATH}: add Coxhead AWL headwords (one word per line)."
        )
    words = [ln.strip() for ln in AWL_HEADWORDS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lower = {w.lower() for w in words}
    stems = {stemmer.stem(w.lower()) for w in words}
    for br, us in _AWL_US_SPELLING.items():
        if br in lower:
            stems.add(stemmer.stem(us))
    return stemmer, frozenset(stems)


def _token_matches_awl(tok: str, stemmer: PorterStemmer, awl_stems: frozenset) -> bool:
    norm = "".join(c for c in tok if c.isalpha())
    if not norm:
        return False
    return stemmer.stem(norm.lower()) in awl_stems


def _text_features(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        return {
            "numeric_token_ratio": float("nan"),
            "nonalpha_token_ratio": float("nan"),
            "long_token_ratio": float("nan"),
            "acronym_ratio": float("nan"),
            "awl_token_ratio": float("nan"),
            "word_count": float("nan"),
        }
    tokens = TOKEN_RE.findall(t)
    if not tokens:
        return {
            "numeric_token_ratio": float("nan"),
            "nonalpha_token_ratio": float("nan"),
            "long_token_ratio": float("nan"),
            "acronym_ratio": float("nan"),
            "awl_token_ratio": float("nan"),
            "word_count": float("nan"),
        }
    n = len(tokens)
    numeric = sum(any(ch.isdigit() for ch in tok) for tok in tokens)
    nonalpha = sum(any(not ch.isalpha() for ch in tok) for tok in tokens)
    longtok = sum(len(tok) >= 10 for tok in tokens)
    acronym = sum(tok.isupper() and len(tok) >= 2 for tok in tokens)
    try:
        stemmer, awl_stems = _awl_stemmer_and_stems()
        awl_hits = sum(_token_matches_awl(tok, stemmer, awl_stems) for tok in tokens)
        awl_ratio = awl_hits / n
    except FileNotFoundError:
        awl_ratio = float("nan")
    return {
        "numeric_token_ratio": numeric / n,
        "nonalpha_token_ratio": nonalpha / n,
        "long_token_ratio": longtok / n,
        "acronym_ratio": acronym / n,
        "awl_token_ratio": awl_ratio,
        "word_count": float(len(t.split())),
    }


def _perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    obs = _spearman(x, y)
    if np.isnan(obs):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        stat = _spearman(x, yp)
        if not np.isnan(stat) and abs(stat) >= abs(obs):
            count += 1
    p = (count + 1.0) / (n_perm + 1.0)
    return float(obs), float(p)


def _center_within_domain(x: np.ndarray, domains: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for d in np.unique(domains):
        idx = domains == d
        out[idx] = x[idx] - np.mean(x[idx])
    return out


def run_text_feature_tests(df_pre: pd.DataFrame, n_perm: int, seed: int) -> List[dict]:
    """
    Test whether numeric / surface-form language proxies are associated with AI scores.
    Includes Coxhead AWL token ratio (Porter stem match to headwords). Uses
    pre-humanization rows only.
    """
    out: List[dict] = []
    if df_pre.empty:
        return out

    # Build one text row per paper/variant/domain from Pangram rows (text source).
    text_base = (
        df_pre[df_pre["detector"] == "pangram"][["collection", "domain", "variant_raw", "paper_id", "text"]]
        .drop_duplicates(subset=["collection", "domain", "variant_raw", "paper_id"])
        .copy()
    )
    if text_base.empty:
        return out

    feats = text_base["text"].map(_text_features).apply(pd.Series)
    text_df = pd.concat([text_base.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    # Limit permutations for this block for runtime control.
    n_perm_feat = min(n_perm, 2000)
    feature_cols = [
        "numeric_token_ratio",
        "nonalpha_token_ratio",
        "long_token_ratio",
        "acronym_ratio",
        "awl_token_ratio",
    ]
    key = ["collection", "domain", "variant_raw", "paper_id"]

    for detector in ["pangram", "gptzero"]:
        det = df_pre[df_pre["detector"] == detector][key + ["score"]].copy()
        merged = det.merge(text_df[key + feature_cols], on=key, how="inner")
        if merged.empty:
            continue
        for feat in feature_cols:
            sub = merged[["domain", "score", feat]].dropna()
            if len(sub) < 30:
                continue
            x = sub[feat].to_numpy(dtype=float)
            y = sub["score"].to_numpy(dtype=float)
            d = sub["domain"].to_numpy()

            rho, p = _perm_spearman(x, y, n_perm=n_perm_feat, seed=seed + len(out) + 7)

            # Domain-adjusted association (center within domain, then Pearson).
            xc = _center_within_domain(x, d)
            yc = _center_within_domain(y, d)
            if np.std(xc) > 0 and np.std(yc) > 0:
                r_adj = float(np.corrcoef(xc, yc)[0, 1])
            else:
                r_adj = float("nan")

            # Top vs bottom quartile contrast for interpretability.
            q1 = float(np.quantile(x, 0.25))
            q3 = float(np.quantile(x, 0.75))
            low = y[x <= q1]
            high = y[x >= q3]
            diff = float(np.mean(high) - np.mean(low)) if low.size and high.size else float("nan")
            # Permutation p-value for high-low mean difference
            if low.size and high.size:
                stat_obs = diff
                combined = np.concatenate([high, low])
                n_h = high.size
                rng = np.random.default_rng(seed + len(out) + 1003)
                count = 0
                for _ in range(n_perm_feat):
                    perm = rng.permutation(combined)
                    stat = float(np.mean(perm[:n_h]) - np.mean(perm[n_h:]))
                    if abs(stat) >= abs(stat_obs):
                        count += 1
                p_hl = (count + 1.0) / (n_perm_feat + 1.0)
            else:
                p_hl = float("nan")

            out.append(
                {
                    "detector": detector,
                    "feature": feat,
                    "n": int(len(sub)),
                    "spearman_rho": rho,
                    "p_value_spearman_permutation": p,
                    "domain_adjusted_r": r_adj,
                    "high_minus_low_q4_q1_mean_score": diff,
                    "p_value_high_low_permutation": p_hl,
                }
            )

    return _apply_fdr_bh(out)


def _apply_fdr_bh(tests: List[dict], alpha: float = 0.05) -> List[dict]:
    """Benjamini--Hochberg FDR adjustment on spearman permutation p-values."""
    if not tests:
        return tests
    ps = [t["p_value_spearman_permutation"] for t in tests]
    m = len(ps)
    order = np.argsort(ps)
    ranked = np.empty(m, dtype=float)
    ranked[order[-1]] = ps[order[-1]]
    for i in range(m - 2, -1, -1):
        idx = order[i]
        ranked[idx] = min(ranked[order[i + 1]], ps[idx] * m / (i + 1))
    for t, q in zip(tests, ranked):
        t["p_value_spearman_fdr_bh"] = float(min(q, 1.0))
        t["significant_fdr_005"] = bool(q < alpha)
    return tests


def pick_example_case(df_pre: pd.DataFrame, df_post: pd.DataFrame) -> Optional[dict]:
    if df_pre.empty or df_post.empty:
        return None
    key_cols = ["collection", "domain", "variant_raw", "paper_id", "detector"]
    merged = (
        df_pre[key_cols + ["score", "text", "path"]]
        .rename(columns={"score": "score_pre", "text": "text_pre", "path": "path_pre"})
        .merge(
            df_post[key_cols + ["score", "text", "path"]].rename(
                columns={"score": "score_post", "text": "text_post", "path": "path_post"}
            ),
            on=key_cols,
            how="inner",
        )
    )
    if merged.empty:
        return None

    # Use Pangram+GPTZero if available for robust cross-detector example.
    core = merged[merged["detector"].isin(["pangram", "gptzero"])].copy()
    if core.empty:
        core = merged.copy()

    group_cols = ["collection", "domain", "variant_raw", "paper_id"]
    agg = (
        core.groupby(group_cols, as_index=False)
        .agg(
            n_detectors=("detector", "nunique"),
            mean_pre=("score_pre", "mean"),
            mean_post=("score_post", "mean"),
        )
        .assign(delta=lambda d: d["mean_post"] - d["mean_pre"])
    )
    if agg.empty:
        return None

    # Prefer pairs with at least 2 detectors, then biggest drop.
    agg = agg.sort_values(["n_detectors", "delta"], ascending=[False, True]).reset_index(drop=True)
    best = agg.iloc[0]

    case_rows = core[
        (core["collection"] == best["collection"])
        & (core["domain"] == best["domain"])
        & (core["variant_raw"] == best["variant_raw"])
        & (core["paper_id"] == best["paper_id"])
    ].copy()

    # Prefer text from Pangram row if present.
    text_row = case_rows[case_rows["detector"] == "pangram"]
    if text_row.empty:
        text_row = case_rows.iloc[[0]]
    text_pre = str(text_row.iloc[0]["text_pre"] or "").strip()
    text_post = str(text_row.iloc[0]["text_post"] or "").strip()

    detector_scores = []
    for _, r in case_rows.sort_values("detector").iterrows():
        detector_scores.append(
            {
                "detector": r["detector"],
                "score_pre": float(r["score_pre"]),
                "score_post": float(r["score_post"]),
                "delta_post_minus_pre": float(r["score_post"] - r["score_pre"]),
                "path_pre": r["path_pre"],
                "path_post": r["path_post"],
            }
        )

    return {
        "collection": best["collection"],
        "domain": best["domain"],
        "variant_raw": best["variant_raw"],
        "variant": VARIANT_LABEL.get(str(best["variant_raw"]), str(best["variant_raw"])),
        "paper_id": best["paper_id"],
        "n_detectors": int(best["n_detectors"]),
        "mean_pre": float(best["mean_pre"]),
        "mean_post": float(best["mean_post"]),
        "delta_post_minus_pre": float(best["delta"]),
        "detector_scores": detector_scores,
        "text_pre": text_pre,
        "text_post": text_post,
    }


def _tex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _truncate_words(text: str, n_words: int = 70) -> str:
    toks = text.split()
    if len(toks) <= n_words:
        return text
    return " ".join(toks[:n_words]) + " ..."


def write_example_tex(example: dict, out_path: Path) -> None:
    rows = []
    for d in example["detector_scores"]:
        rows.append(
            f"{_tex_escape(d['detector'])} & {d['score_pre']:.3f} & {d['score_post']:.3f} & {d['delta_post_minus_pre']:.3f} \\\\"
        )
    rows_text = "\n".join(rows)
    pre_short = _tex_escape(_truncate_words(example.get("text_pre", ""), 70))
    post_short = _tex_escape(_truncate_words(example.get("text_post", ""), 70))
    variant = _tex_escape(example["variant"])
    domain = _tex_escape(str(example["domain"]).replace("_", " "))
    paper_id = _tex_escape(example["paper_id"])
    snippet = f"""
\\paragraph{{Auto-selected example case.}}
Collection: \\texttt{{{_tex_escape(example['collection'])}}}, domain: \\texttt{{{domain}}}, variant: \\texttt{{{variant}}}, paper ID: \\texttt{{{paper_id}}}.
This case is selected by largest cross-detector score drop after humanization.

\\begin{{table}}[h]
\\centering
\\caption{{Detector score change for selected example (post - pre).}}
\\begin{{tabular}}{{lrrr}}
\\hline
Detector & Pre & Post & $\\Delta$ \\\\
\\hline
{rows_text}
\\hline
\\end{{tabular}}
\\end{{table}}

\\textbf{{Pre-humanization excerpt:}} {pre_short}

\\textbf{{Post-humanization excerpt:}} {post_short}
""".strip()
    out_path.write_text(snippet + "\n", encoding="utf-8")


def write_summary_markdown(
    out_path: Path,
    humanization_tests: List[dict],
    domain_tests: List[dict],
    variant_tests: List[dict],
    agreement_tests: List[dict],
    text_feature_tests: List[dict],
    example: Optional[dict],
) -> None:
    lines: List[str] = []
    lines.append("# Statistical Summary for Paper")
    lines.append("")
    lines.append("## Humanization Effect (matched pre vs post)")
    for t in humanization_tests:
        lines.append(
            "- detector={detector}, n={n_pairs}, mean_pre={mean_pre:.4f}, mean_post={mean_post:.4f}, "
            "delta={mean_diff_post_minus_pre:.4f}, p={p_value_permutation_two_sided:.4g}".format(**t)
        )
    lines.append("")
    lines.append("## Domain Effect (non-STEM minus STEM)")
    for t in domain_tests:
        lines.append(
            "- phase={phase}, detector={detector}, n_non_stem={n_non_stem}, n_stem={n_stem}, "
            "mean_diff={mean_non_stem_minus_stem:.4f}, p={p_value_permutation_two_sided:.4g}".format(**t)
        )
    lines.append("")
    lines.append("## Variant Effect (one-way permutation ANOVA)")
    for t in variant_tests:
        lines.append(
            "- phase={phase}, detector={detector}, n={n_total}, F={f_statistic:.4f}, p={p_value_permutation:.4g}".format(**t)
        )
    lines.append("")
    lines.append("## Pangram vs GPTZero Agreement")
    for t in agreement_tests:
        lines.append(
            "- phase={phase}, n={n_pairs}, spearman={spearman_rho:.4f}, kappa={cohen_kappa_thresholded:.4f}, "
            "threshold={threshold:.2f}".format(**t)
        )
    lines.append("")
    lines.append("## Text Feature Associations (Pre-Humanization)")
    for t in text_feature_tests:
        lines.append(
            "- detector={detector}, feature={feature}, n={n}, spearman={spearman_rho:.4f}, "
            "p={p_value_spearman_permutation:.4g}, domain_adj_r={domain_adjusted_r:.4f}, "
            "high-low={high_minus_low_q4_q1_mean_score:.4f}, p_high-low={p_value_high_low_permutation:.4g}".format(**t)
        )
    if example:
        lines.append("")
        lines.append("## Auto-selected Example Case")
        lines.append(
            "- collection={collection}, domain={domain}, variant={variant}, paper_id={paper_id}, "
            "mean_pre={mean_pre:.4f}, mean_post={mean_post:.4f}, delta={delta_post_minus_pre:.4f}".format(**example)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_insert_text(
    out_path: Path,
    humanization_tests: List[dict],
    domain_tests: List[dict],
    variant_tests: List[dict],
    agreement_tests: List[dict],
    text_feature_tests: List[dict],
    example: Optional[dict],
) -> None:
    """
    Write plain text that can be copied directly into paper prose.
    """
    lines: List[str] = []
    lines.append("Paper-Ready Numeric Results")
    lines.append("")
    lines.append("Humanization effect (matched pre vs post):")
    for t in humanization_tests:
        lines.append(
            (
                f"- {t['detector']}: n={t['n_pairs']}, mean_pre={t['mean_pre']:.4f}, "
                f"mean_post={t['mean_post']:.4f}, delta={t['mean_diff_post_minus_pre']:.4f}, "
                f"p={t['p_value_permutation_two_sided']:.4g}"
            )
        )
    lines.append("")
    lines.append("Domain effect (non-STEM minus STEM):")
    for t in domain_tests:
        lines.append(
            (
                f"- phase={t['phase']}, detector={t['detector']}, n_non_stem={t['n_non_stem']}, "
                f"n_stem={t['n_stem']}, mean_diff={t['mean_non_stem_minus_stem']:.4f}, "
                f"p={t['p_value_permutation_two_sided']:.4g}"
            )
        )
    lines.append("")
    lines.append("Variant effect (one-way permutation ANOVA):")
    for t in variant_tests:
        lines.append(
            (
                f"- phase={t['phase']}, detector={t['detector']}, n={t['n_total']}, "
                f"F={t['f_statistic']:.4f}, p={t['p_value_permutation']:.4g}"
            )
        )
    lines.append("")
    lines.append("Detector agreement (Pangram vs GPTZero):")
    for t in agreement_tests:
        lines.append(
            (
                f"- phase={t['phase']}, n={t['n_pairs']}, spearman={t['spearman_rho']:.4f}, "
                f"kappa={t['cohen_kappa_thresholded']:.4f} at threshold={t['threshold']:.2f}"
            )
        )
    lines.append("")
    lines.append("Text-feature association tests (pre-humanization):")
    for t in text_feature_tests:
        lines.append(
            (
                f"- detector={t['detector']}, feature={t['feature']}, n={t['n']}, "
                f"spearman={t['spearman_rho']:.4f}, p={t['p_value_spearman_permutation']:.4g}, "
                f"domain_adj_r={t['domain_adjusted_r']:.4f}, "
                f"high_minus_low={t['high_minus_low_q4_q1_mean_score']:.4f}, "
                f"p_high_low={t['p_value_high_low_permutation']:.4g}"
            )
        )
    if example:
        lines.append("")
        lines.append("Auto-selected example case:")
        lines.append(
            (
                f"- collection={example['collection']}, domain={example['domain']}, "
                f"variant={example['variant']}, paper_id={example['paper_id']}, "
                f"mean_pre={example['mean_pre']:.4f}, mean_post={example['mean_post']:.4f}, "
                f"delta={example['delta_post_minus_pre']:.4f}"
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_for_collection(collection: str, n_perm: int, threshold: float, seed: int) -> None:
    out_dir = OUT_BASE / collection
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pre = load_rows(collection, phase="pre")
    df_post = load_rows(collection, phase="post")

    if df_pre.empty:
        raise FileNotFoundError(f"No pre-humanization rows found under {PRE_BASE / collection}")
    if df_post.empty:
        raise FileNotFoundError(f"No post-humanization rows found under {POST_BASE / collection}")

    df_pre.to_csv(out_dir / "rows_pre.csv", index=False)
    df_post.to_csv(out_dir / "rows_post.csv", index=False)
    export_pangram_summaries(out_dir, df_pre, df_post)

    df_all = pd.concat([df_pre, df_post], ignore_index=True)

    humanization_tests = run_humanization_tests(df_pre, df_post, n_perm=n_perm, seed=seed)
    domain_tests = run_domain_tests(df_all, n_perm=n_perm, seed=seed + 123)
    variant_tests = run_variant_effect_tests(df_all, n_perm=n_perm, seed=seed + 321)
    agreement_tests = run_agreement_tests(df_all, threshold=threshold)
    text_feature_tests = run_text_feature_tests(df_pre, n_perm=n_perm, seed=seed + 777)
    example = pick_example_case(df_pre, df_post)

    (out_dir / "tests_humanization.json").write_text(
        json.dumps(humanization_tests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tests_domain_stem_vs_nonstem.json").write_text(
        json.dumps(domain_tests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tests_variant_effect.json").write_text(
        json.dumps(variant_tests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tests_detector_agreement.json").write_text(
        json.dumps(agreement_tests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tests_text_features.json").write_text(
        json.dumps(text_feature_tests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if example:
        (out_dir / "example_case.json").write_text(
            json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_example_tex(example, out_dir / "example_snippet.tex")

    write_summary_markdown(
        out_dir / "summary_for_paper.md",
        humanization_tests=humanization_tests,
        domain_tests=domain_tests,
        variant_tests=variant_tests,
        agreement_tests=agreement_tests,
        text_feature_tests=text_feature_tests,
        example=example,
    )
    write_paper_insert_text(
        out_dir / "paper_insert.txt",
        humanization_tests=humanization_tests,
        domain_tests=domain_tests,
        variant_tests=variant_tests,
        agreement_tests=agreement_tests,
        text_feature_tests=text_feature_tests,
        example=example,
    )

    print(f"Collection: {collection}")
    print(f"  Pre rows: {len(df_pre)}")
    print(f"  Post rows: {len(df_post)}")
    print(f"  Humanization tests: {len(humanization_tests)}")
    print(f"  Domain tests: {len(domain_tests)}")
    print(f"  Variant tests: {len(variant_tests)}")
    print(f"  Agreement tests: {len(agreement_tests)}")
    print(f"  Text feature tests: {len(text_feature_tests)}")
    if example:
        print(
            f"  Example: {example['paper_id']} ({example['domain']}/{example['variant_raw']}) "
            f"delta={example['delta_post_minus_pre']:.4f}"
        )
    print(f"  Wrote: {out_dir}")

    try:
        import sys

        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from robustness_analysis import run_collection as run_robustness

        run_robustness(collection, tau=threshold, n_boot=2000, seed=seed)
    except Exception as exc:
        print(f"  Robustness analysis skipped: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper statistical tests and pick one example.")
    parser.add_argument(
        "--collections",
        nargs="*",
        default=["2015_back_2013", "2025_back_2023"],
        help="Collections to analyze (default: both).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=5000,
        help="Permutation iterations per test (default: 5000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="AI-positive threshold for kappa agreement (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for c in args.collections:
        run_for_collection(c, n_perm=args.n_perm, threshold=args.threshold, seed=args.seed)


if __name__ == "__main__":
    main()

