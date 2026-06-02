#!/usr/bin/env python3
"""
Policy-oriented figures for the paper (error modes, not classifier benchmarking).

Replaces precision-recall curves as the primary robustness visualization:
  - flag_rates_by_condition.png: flag rate by rewrite condition (RQ1 / assisted-editing risk)
  - domain_flag_rates.png: original flag rate by domain (equity / differential impact)
  - evasion_pre_post_fnr.png: pre vs post FNR on AI-labeled variants (RQ3)
  - humanization_feature_shift.png: paired linguistic shifts after Undetectable (mechanism)

Usage:
  PYTHONPATH=src python3 src/plot_policy_figures.py --collection 2025_back_2023
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_stats import VARIANT_LABEL, load_rows
from variants import VARIANTS

AI_VARIANTS = tuple(v for v in VARIANTS if v != "original")
# Each curve: original = negative, one positive rewrite type (or pooled AI).
REWRITE_CONDITION_TASKS = (
    ("refine_abstract_only", "Refine (abs. only)"),
    ("refine_abstract_article", "Refine (abs. + article)"),
    ("new_article_only", "New (article only)"),
    ("pooled_ai", "Pooled AI (all rewrites)"),
)
CONDITION_COLORS = {
    "refine_abstract_only": "#2563eb",
    "refine_abstract_article": "#7c3aed",
    "new_article_only": "#dc2626",
    "pooled_ai": "#64748b",
}
COMMERCIAL_DETECTORS = ("pangram", "gptzero")

ROOT = Path(__file__).resolve().parents[1]
HUMANIZATION_DIR = ROOT / "humanization"
FIGURES_BASE = ROOT / "results" / "figures"

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

VARIANT_ORDER = VARIANTS
VARIANT_DISPLAY = [VARIANT_LABEL[v] for v in VARIANT_ORDER]
DOMAIN_ORDER = ("chemistry", "computer_science", "political_science", "theology")
DOMAIN_SHORT = {
    "chemistry": "Chemistry",
    "computer_science": "Comp. sci.",
    "political_science": "Pol. sci.",
    "theology": "Theology",
}
TAU = 0.5
COLORS = {"pangram": "#2563eb", "gptzero": "#dc2626", "llm_assisted": "#16a34a"}
# Assisted-editing task: original = negative, refine (abs. only) = positive
ASSISTED_DETECTORS = (
    ("pangram", "Pangram"),
    ("gptzero", "GPTZero"),
    ("llm_assisted", "LLM-assisted"),
)


def _period_label(collection: str) -> str:
    return "2013–2015" if "2013" in collection else "2023–2025"


def _is_pre_llm_collection(collection: str) -> bool:
    return "2013" in collection


def _original_negative_axis_label(collection: str, *, short: bool = False) -> str:
    """ROC x-axis: proxy FPR (2013–2015) vs flag rate on originals (2023–2025)."""
    if _is_pre_llm_collection(collection):
        return "FPR on originals" if short else "False positive rate on originals"
    return "Flag rate on originals"


def _flag_rate(scores: np.ndarray, tau: float = TAU) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores >= tau))


def _fnr_rate(scores: np.ndarray, tau: float = TAU) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores < tau))


def plot_flag_rates_by_condition(df_pre: pd.DataFrame, collection: str, out: Path) -> None:
    """% flagged at tau by rewrite condition — core RQ1 policy figure."""
    sub = df_pre[(df_pre["collection"] == collection) & (df_pre["detector"].isin(("pangram", "gptzero")))]
    if sub.empty:
        return

    x = np.arange(len(VARIANT_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for i, det in enumerate(("pangram", "gptzero")):
        rates = []
        for var in VARIANT_ORDER:
            s = sub[(sub["detector"] == det) & (sub["variant_raw"] == var)]["score"].to_numpy(dtype=float)
            rates.append(100 * _flag_rate(s))
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            rates,
            width,
            label=det.capitalize(),
            color=COLORS[det],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Original", "Refine\n(abs. only)", "Refine\n(abs.+article)", "New\n(article only)"],
        fontsize=9,
    )
    ax.set_ylabel(f"Flag rate at $\\tau={TAU}$ (%)")
    ax.set_ylim(0, 105)
    ax.axhline(TAU * 100, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.legend(loc="upper left", frameon=True)
    period = _period_label(collection)
    ax.set_title(f"Flag rates by rewrite condition ({period}, pre-humanization)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_domain_flag_rates(df_pre: pd.DataFrame, collection: str, out: Path) -> None:
    """Original vs refine (abs. only) flag rates by domain — differential impact."""
    sub = df_pre[
        (df_pre["collection"] == collection)
        & (df_pre["detector"].isin(("pangram", "gptzero")))
        & (df_pre["variant_raw"].isin(("original", "refine_abstract_only")))
    ]
    if sub.empty:
        return

    x = np.arange(len(DOMAIN_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.2))

    combos = [
        ("pangram", "original", "Pangram / original"),
        ("pangram", "refine_abstract_only", "Pangram / refine (abs. only)"),
        ("gptzero", "original", "GPTZero / original"),
        ("gptzero", "refine_abstract_only", "GPTZero / refine (abs. only)"),
    ]
    for i, (det, var, label) in enumerate(combos):
        rates = []
        for dom in DOMAIN_ORDER:
            s = sub[
                (sub["detector"] == det)
                & (sub["variant_raw"] == var)
                & (sub["domain"] == dom)
            ]["score"].to_numpy(dtype=float)
            rates.append(100 * _flag_rate(s))
        offset = (i - 1.5) * width
        color = COLORS[det]
        alpha = 1.0 if var == "original" else 0.55
        hatch = "" if var == "original" else "//"
        ax.bar(
            x + offset,
            rates,
            width,
            label=label,
            color=color,
            alpha=alpha,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER])
    ax.set_ylabel(f"Flag rate at $\\tau={TAU}$ (%)")
    ax.set_ylim(0, 100)
    period = _period_label(collection)
    ax.set_title(f"Domain-level flag rates ({period}, pre-humanization)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_evasion_pre_post_fnr(
    df_pre: pd.DataFrame, df_post: pd.DataFrame, collection: str, out: Path
) -> None:
    """Pre vs post FNR on pooled AI-labeled variants (RQ3)."""
    ai_vars = AI_VARIANTS
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    width = 0.32

    for i, det in enumerate(("pangram", "gptzero")):
        pre = df_pre[
            (df_pre["collection"] == collection)
            & (df_pre["detector"] == det)
            & (df_pre["variant_raw"].isin(ai_vars))
        ]["score"].to_numpy(dtype=float)
        post = df_post[
            (df_post["collection"] == collection)
            & (df_post["detector"] == det)
            & (df_post["variant_raw"].isin(ai_vars))
        ]["score"].to_numpy(dtype=float)
        vals = [100 * _fnr_rate(pre), 100 * _fnr_rate(post)]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=det.capitalize(),
            color=COLORS[det],
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-humanization", "Post-humanization"])
    ax.set_ylabel(f"FNR at $\\tau={TAU}$ on AI-labeled rewrites (%)")
    ax.set_ylim(0, 105)
    period = _period_label(collection)
    ax.set_title(f"Evasion: false negatives after humanization ({period})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_humanization_feature_means(collection: str) -> dict[str, float]:
    """Pooled pre/post means for key features from humanization/ JSON."""
    import json

    keys = ("long_token_ratio", "awl_token_ratio", "type_token_ratio")
    pre_acc: dict[str, list] = {k: [] for k in keys}
    post_acc: dict[str, list] = {k: [] for k in keys}
    base = HUMANIZATION_DIR / collection
    if not base.is_dir():
        return {}

    for dom_dir in base.iterdir():
        if not dom_dir.is_dir():
            continue
        for var in VARIANT_ORDER:
            vdir = dom_dir / var
            if not vdir.is_dir():
                continue
            for p in vdir.glob("W*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                pre_t = data.get("original_abstract") or ""
                post_t = data.get("humanized_abstract") or ""
                if not pre_t or not post_t:
                    continue
                pre_f = _extended_text_features(pre_t)
                post_f = _extended_text_features(post_t)
                for k in keys:
                    if np.isfinite(pre_f.get(k, float("nan"))):
                        pre_acc[k].append(pre_f[k])
                    if np.isfinite(post_f.get(k, float("nan"))):
                        post_acc[k].append(post_f[k])

    out: dict[str, float] = {}
    for k in keys:
        if pre_acc[k]:
            out[f"pre_{k}"] = float(np.mean(pre_acc[k]))
        if post_acc[k]:
            out[f"post_{k}"] = float(np.mean(post_acc[k]))
    return out


def _extended_text_features(text: str) -> dict:
    from paper_stats import _text_features

    feats = _text_features(text)
    tokens = __import__("re").findall(r"[A-Za-z0-9][A-Za-z0-9\-\+\./]*", (text or "").strip())
    n = len(tokens)
    feats["type_token_ratio"] = len({t.lower() for t in tokens}) / n if n else float("nan")
    return feats


def _assisted_editing_subset(df_pre: pd.DataFrame, collection: str) -> pd.DataFrame:
    return df_pre[
        (df_pre["collection"] == collection)
        & (df_pre["detector"].isin([d[0] for d in ASSISTED_DETECTORS]))
        & (df_pre["variant_raw"].isin(("original", "refine_abstract_only")))
    ]


def _roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC curve; returns FPR, TPR, AUC-ROC (positive class = 1)."""
    if y_true.size == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-scores)
    y = y_true[order].astype(int)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    tpr = np.concatenate([[0.0], tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / n_neg, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _operating_point(y_true: np.ndarray, scores: np.ndarray, tau: float) -> dict:
    flagged = scores >= tau
    y = y_true.astype(int)
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum((~flagged) & (y == 1)))
    tn = int(np.sum((~flagged) & (y == 0)))
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    return {
        "tpr": tp / n_pos,
        "fpr": fp / n_neg,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / n_pos,
    }


def _pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Precision-recall curve; returns P, R, AUC-PR."""
    if y_true.size == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-scores)
    y = y_true[order].astype(int)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    n_pos = max(int(y.sum()), 1)
    recall = tp / n_pos
    prec = np.concatenate([[1.0], precision, [precision[-1] if precision.size else 0.0]])
    rec = np.concatenate([[0.0], recall, [1.0]])
    auc = float(np.trapezoid(prec, rec))
    return prec, rec, auc


def _plot_assisted_roc(
    sub: pd.DataFrame,
    ax: plt.Axes,
    collection: str,
    *,
    panel: bool = False,
    show_legend: bool = True,
    legend_fontsize: float = 7,
) -> None:
    for det_key, det_label in ASSISTED_DETECTORS:
        d = sub[sub["detector"] == det_key]
        if d.empty:
            continue
        y = np.where(d["variant_raw"].to_numpy() == "refine_abstract_only", 1, 0)
        scores = d["score"].to_numpy(dtype=float)
        fpr, tpr, auc_roc = _roc_curve(y, scores)
        if fpr.size == 0:
            continue
        ax.plot(
            fpr,
            tpr,
            label=f"{det_label} (AUC={auc_roc:.2f})",
            color=COLORS[det_key],
            linewidth=2,
        )
        op = _operating_point(y, scores, TAU)
        ax.scatter(
            [op["fpr"]],
            [op["tpr"]],
            color=COLORS[det_key],
            s=36,
            zorder=5,
            edgecolors="white",
            linewidths=0.7,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=1)
    ax.set_xlabel(_original_negative_axis_label(collection))
    ax.set_ylabel("True positive rate on refine (abs. only)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) ROC" if panel else "ROC")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=legend_fontsize, frameon=True)


def _plot_assisted_pr(
    sub: pd.DataFrame,
    ax: plt.Axes,
    *,
    panel: bool = False,
    show_legend: bool = True,
    legend_fontsize: float = 7,
) -> None:
    for det_key, det_label in ASSISTED_DETECTORS:
        d = sub[sub["detector"] == det_key]
        if d.empty:
            continue
        y = np.where(d["variant_raw"].to_numpy() == "refine_abstract_only", 1, 0)
        scores = d["score"].to_numpy(dtype=float)
        prec, rec, auc_pr = _pr_curve(y, scores)
        if rec.size == 0:
            continue
        ax.plot(
            rec,
            prec,
            label=f"{det_label} (AUC-PR={auc_pr:.2f})",
            color=COLORS[det_key],
            linewidth=2,
        )
        op = _operating_point(y, scores, TAU)
        ax.scatter(
            [op["recall"]],
            [op["precision"]],
            color=COLORS[det_key],
            s=36,
            zorder=5,
            edgecolors="white",
            linewidths=0.7,
        )
    ax.set_xlabel("Recall on refine (abs. only)")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) Precision-recall" if panel else "Precision-recall")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="lower left", fontsize=legend_fontsize, frameon=True)


def plot_assisted_editing_roc(df_pre: pd.DataFrame, collection: str, out: Path) -> None:
    """Single ROC panel (main text); compact aspect for wrapfigure."""
    sub = _assisted_editing_subset(df_pre, collection)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    _plot_assisted_roc(
        sub, ax, collection, panel=False, show_legend=True, legend_fontsize=6
    )
    period = _period_label(collection)
    ax.set_title(f"Assisted editing ({period})", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_assisted_editing_roc_pr_panel(df_pre: pd.DataFrame, collection: str, out: Path) -> None:
    """Side-by-side ROC + PR (appendix; all detectors, assisted-editing task)."""
    sub = _assisted_editing_subset(df_pre, collection)
    if sub.empty:
        return
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    _plot_assisted_roc(
        sub, ax_roc, collection, panel=True, show_legend=True, legend_fontsize=6.5
    )
    _plot_assisted_pr(sub, ax_pr, panel=True, show_legend=False)
    period = _period_label(collection)
    fig.suptitle(f"Assisted-editing detection ({period}, pre-humanization)", y=1.02, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _original_vs_positive_rows(
    df_pre: pd.DataFrame, collection: str, detector: str, positive: str
) -> tuple[np.ndarray, np.ndarray]:
    """Binary labels: original=0, positive rewrite(s)=1."""
    base = df_pre[
        (df_pre["collection"] == collection) & (df_pre["detector"] == detector)
    ]
    if positive == "pooled_ai":
        sub = base[base["variant_raw"].isin(("original",) + AI_VARIANTS)]
        y = np.where(sub["variant_raw"].to_numpy() == "original", 0, 1)
    else:
        sub = base[base["variant_raw"].isin(("original", positive))]
        y = np.where(sub["variant_raw"].to_numpy() == positive, 1, 0)
    return y, sub["score"].to_numpy(dtype=float)


def _plot_condition_curves_on_axes(
    df_pre: pd.DataFrame,
    collection: str,
    detector: str,
    ax_roc: plt.Axes,
    ax_pr: plt.Axes,
) -> None:
    for pos_key, pos_label in REWRITE_CONDITION_TASKS:
        y, scores = _original_vs_positive_rows(df_pre, collection, detector, pos_key)
        if y.size == 0:
            continue
        linestyle = "--" if pos_key == "pooled_ai" else "-"
        color = CONDITION_COLORS[pos_key]
        fpr, tpr, auc_roc = _roc_curve(y, scores)
        prec, rec, auc_pr = _pr_curve(y, scores)
        short = pos_label.replace(" (", "\n(") if len(pos_label) > 22 else pos_label
        if fpr.size:
            ax_roc.plot(
                fpr,
                tpr,
                linestyle=linestyle,
                color=color,
                linewidth=2 if pos_key != "pooled_ai" else 1.5,
                label=f"{short} (AUC {auc_roc:.2f})",
            )
            op = _operating_point(y, scores, TAU)
            ax_roc.scatter(
                [op["fpr"]],
                [op["tpr"]],
                color=color,
                s=28,
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )
        if rec.size:
            ax_pr.plot(
                rec,
                prec,
                linestyle=linestyle,
                color=color,
                linewidth=2 if pos_key != "pooled_ai" else 1.5,
                label=f"{short} (AUC-PR {auc_pr:.2f})",
            )
            op = _operating_point(y, scores, TAU)
            ax_pr.scatter(
                [op["recall"]],
                [op["precision"]],
                color=color,
                s=28,
                zorder=5,
                edgecolors="white",
                linewidths=0.6,
            )
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=0.8)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(True, alpha=0.3)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.grid(True, alpha=0.3)


def plot_rewrite_condition_roc_pr(
    df_pre: pd.DataFrame, collection: str, out: Path
) -> None:
    """
    Appendix: ROC/PR per rewrite condition vs. original (Pangram and GPTZero).
    Four curves per panel: three single-condition tasks plus pooled AI rewrites.
    """
    period = _period_label(collection)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    det_titles = {"pangram": "Pangram", "gptzero": "GPTZero"}
    for row, det in enumerate(COMMERCIAL_DETECTORS):
        _plot_condition_curves_on_axes(
            df_pre, collection, det, axes[row, 0], axes[row, 1]
        )
        axes[row, 0].set_ylabel(f"{det_titles[det]}\nTPR on rewrite")
        axes[row, 0].legend(loc="lower right", fontsize=6.2, frameon=True)
        if row == 0:
            roc_title = "ROC (original = negative class)"
            if not _is_pre_llm_collection(collection):
                roc_title = "ROC (original = negative; x-axis = flag rate)"
            axes[row, 0].set_title(roc_title)
            axes[row, 1].set_title("Precision-recall")
        axes[row, 1].legend(loc="lower left", fontsize=6.2, frameon=True)
    x_label = _original_negative_axis_label(collection, short=True)
    for ax in axes[:, 0]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 1]:
        ax.set_xlabel("Recall on rewrite")
        ax.set_ylabel("Precision")
    fig.suptitle(
        f"Detection by rewrite condition vs. original ({period}, pre-humanization)",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_assisted_editing_pr(df_pre: pd.DataFrame, collection: str, out: Path) -> None:
    """Standalone PR (appendix); same detectors and task as the main-text ROC|PR panel."""
    sub = _assisted_editing_subset(df_pre, collection)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _plot_assisted_pr(sub, ax, show_legend=True, legend_fontsize=8)
    ax.set_title(f"Assisted-editing PR ({_period_label(collection)}, pre-humanization)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(df_pre: pd.DataFrame, collection: str, out: Path) -> None:
    """Policy trade-off as tau varies: original flag rate, refine flagged rate, AI-labeled FNR."""
    ai_vars = AI_VARIANTS
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    for ax, det in zip(axes, ("pangram", "gptzero")):
        orig_rates, refine_rates, fnr_rates = [], [], []
        sub = df_pre[(df_pre["collection"] == collection) & (df_pre["detector"] == det)]
        for tau in DEFAULT_THRESHOLDS:
            orig = sub[sub["variant_raw"] == "original"]["score"].to_numpy(dtype=float)
            refine = sub[sub["variant_raw"] == "refine_abstract_only"]["score"].to_numpy(dtype=float)
            ai = sub[sub["variant_raw"].isin(ai_vars)]["score"].to_numpy(dtype=float)
            orig_rates.append(100 * _flag_rate(orig, tau))
            refine_rates.append(100 * _flag_rate(refine, tau))
            fnr_rates.append(100 * _fnr_rate(ai, tau))

        ax.plot(DEFAULT_THRESHOLDS, orig_rates, "o-", color="#64748b", label="Original flag rate")
        ax.plot(DEFAULT_THRESHOLDS, refine_rates, "s-", color=COLORS[det], label="Refine (abs. only) flagged")
        ax.plot(DEFAULT_THRESHOLDS, fnr_rates, "^--", color="#b45309", label="FNR (AI-labeled pool)")
        ax.axvline(TAU, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Threshold $\\tau$")
        ax.set_title(det.capitalize())
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Rate (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    period = _period_label(collection)
    fig.suptitle(f"Threshold sensitivity ({period}, pre-humanization)", y=1.02, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_humanization_feature_shift(collection: str, out: Path) -> None:
    """Paired pre/post means for detector-salient features (mechanism)."""
    means = _load_humanization_feature_means(collection)
    if not means:
        return

    labels = ["Long-token\nratio", "AWL token\nratio", "Type-token\nratio"]
    keys = ["long_token_ratio", "awl_token_ratio", "type_token_ratio"]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))

    pre_vals = [means.get(f"pre_{k}", 0) for k in keys]
    post_vals = [means.get(f"post_{k}", 0) for k in keys]
    ax.bar(x - width / 2, pre_vals, width, label="Before humanization", color="#94a3b8")
    ax.bar(x + width / 2, post_vals, width, label="After humanization", color="#0f766e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean ratio")
    ax.set_ylim(0, max(pre_vals + post_vals) * 1.15)
    period = _period_label(collection)
    ax.set_title(f"Linguistic shifts after Undetectable v11 ({period}, pooled)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="2015_back_2013")
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="If omitted, only --collection is run. Pass both period dirs for full paper figures.",
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_BASE)
    args = parser.parse_args()
    collections = args.collections or [args.collection]

    for collection in collections:
        df_pre = load_rows(collection, "pre")
        df_post = load_rows(collection, "post")
        out_dir = args.figures_dir / collection
        plot_flag_rates_by_condition(
            df_pre, collection, out_dir / "flag_rates_by_condition.png"
        )
        plot_domain_flag_rates(df_pre, collection, out_dir / "domain_flag_rates.png")
        plot_evasion_pre_post_fnr(
            df_pre, df_post, collection, out_dir / "evasion_pre_post_fnr.png"
        )
        plot_humanization_feature_shift(
            collection, out_dir / "humanization_feature_shift.png"
        )
        plot_assisted_editing_roc(
            df_pre, collection, out_dir / "assisted_editing_roc.png"
        )
        plot_assisted_editing_roc_pr_panel(
            df_pre, collection, out_dir / "assisted_editing_roc_pr.png"
        )
        plot_rewrite_condition_roc_pr(
            df_pre, collection, out_dir / "rewrite_condition_roc_pr.png"
        )
        plot_threshold_tradeoff(
            df_pre, collection, out_dir / "threshold_tradeoff.png"
        )
        print(f"Wrote policy figures under {out_dir}")


if __name__ == "__main__":
    main()
