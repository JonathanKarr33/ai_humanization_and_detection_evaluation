#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plot_pangram_vs_gptzero_agreement import load_pairs, plot_scatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTIONS = ["2025_back_2023", "2015_back_2013"]

CANONICAL_PROCESS_ORDER = ["original", "rewritten", "improved", "new"]
PROCESS_LABELS = {
    "original": "original",
    "rewritten": "polish",
    "improved": "refine",
    "new": "new",
}


def canonical_process(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"polish", "rewritten"}:
        return "rewritten"
    if v in {"refine", "improve", "improved"}:
        return "improved"
    if v in {"original", "new"}:
        return v
    return v


def display_process(value: str) -> str:
    return PROCESS_LABELS.get(value, value)


def discover_collections(papers_dir: Path) -> list[str]:
    collections: list[str] = []
    if not papers_dir.exists():
        return collections
    for p in sorted(papers_dir.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        collections.append(p.name)
    return collections


def default_collections(papers_dir: Path) -> list[str]:
    available = set(discover_collections(papers_dir))
    preferred = [c for c in DEFAULT_COLLECTIONS if c in available]
    if preferred:
        return preferred
    return sorted(available)


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


def load_records_for_collection(collection: str) -> list[dict]:
    meta_path = ROOT / "papers" / collection / f"metadata_{collection}.jsonl"
    if meta_path.exists():
        records = load_metadata_jsonl(meta_path)
        if records:
            return records

    # Fallback: derive records from paper_jsons when metadata_{collection}.jsonl is absent.
    records: list[dict] = []
    base = ROOT / "papers" / collection
    if not base.exists():
        return records
    for dom_dir in sorted(base.iterdir(), key=lambda p: p.name):
        if not dom_dir.is_dir():
            continue
        paper_jsons = dom_dir / "paper_jsons"
        if not paper_jsons.exists():
            continue
        for p in sorted(paper_jsons.glob("W*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            records.append(
                {
                    "domain": obj.get("domain") or dom_dir.name,
                    "abstract": obj.get("abstract") or "",
                    "publication_date": obj.get("publication_date"),
                    "publication_month": obj.get("publication_month"),
                }
            )
    return records


def render_abstract_coverage(collection: str, min_words: int, outdir: Path) -> Path:
    records = load_records_for_collection(collection)
    if not records:
        raise RuntimeError(f"No records found for collection: {collection}")

    filtered = []
    for rec in records:
        if not rec.get("domain"):
            continue
        if word_count(rec.get("abstract") or "") < min_words:
            continue
        filtered.append(rec)
    if not filtered:
        raise RuntimeError(f"No abstracts >= {min_words} words found in {meta_path}")

    rows = [{"domain": rec.get("domain"), "month": normalize_month(rec)} for rec in filtered]
    df = pd.DataFrame(rows)

    domain_counts = df["domain"].value_counts().sort_values(ascending=False)
    df_month = df.dropna(subset=["month"]).copy()
    month_pivot = (
        df_month.pivot_table(index="month", columns="domain", aggfunc="size", fill_value=0).sort_index()
    )

    if len(month_pivot) > 0:
        start = month_pivot.index.min()
        end = month_pivot.index.max()
        full = pd.period_range(start=start, end=end, freq="M").astype(str)
        month_pivot = month_pivot.reindex(full, fill_value=0)

    final_outdir = outdir / collection
    final_outdir.mkdir(parents=True, exist_ok=True)
    outpath = final_outdir / f"abstract_counts_min{min_words}.png"

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.65)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(domain_counts.index.tolist(), domain_counts.values.tolist(), color="#4c78a8")
    ax1.set_title(f"{collection}: abstracts >= {min_words} words - count by domain", pad=10)
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
        ax2.set_title(f"{collection}: abstracts >= {min_words} words - count by month (stacked)", pad=12)
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

    fig.suptitle(f"Abstract coverage - {collection}", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.93, right=0.80)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ai_likelihood" in df.columns:
        df["ai_likelihood"] = pd.to_numeric(df["ai_likelihood"], errors="coerce")
    for col in ("domain", "process", "paper_id"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "process" in df.columns:
        df["process_canonical"] = df["process"].fillna("").map(canonical_process)
    return df


def process_order(df: pd.DataFrame) -> List[str]:
    if "process_canonical" not in df.columns:
        return []
    present = [p for p in CANONICAL_PROCESS_ORDER if (df["process_canonical"] == p).any()]
    extras = sorted(
        [p for p in df["process_canonical"].dropna().unique().tolist() if p not in CANONICAL_PROCESS_ORDER]
    )
    return present + extras


def flierprops() -> dict:
    return {
        "marker": "o",
        "markersize": 3.5,
        "markerfacecolor": "crimson",
        "markeredgecolor": "crimson",
        "alpha": 0.55,
    }


def count_outliers_iqr(values: List[float]) -> int:
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
        lower = xs[: n // 2]
        upper = xs[n // 2 :]
    else:
        lower = xs[: n // 2]
        upper = xs[n // 2 + 1 :]

    if not lower or not upper:
        return 0

    q1 = median(lower)
    q3 = median(upper)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return sum(1 for v in xs if v < low or v > high)


def render_results_boxplot(input_csv: Path, outdir: Path, max_domains: int = 16) -> Path:
    df = load_results(input_csv)
    if "ai_likelihood" not in df.columns or "domain" not in df.columns or "process_canonical" not in df.columns:
        raise RuntimeError("results CSV is missing one of: ai_likelihood, domain, process")

    sns.set_theme(style="whitegrid")
    proc_order = process_order(df)
    if not proc_order:
        raise RuntimeError("No process values found in results CSV.")

    dom_counts = df["domain"].value_counts()
    domains = dom_counts.head(max_domains).index.tolist()

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
        sub = df[(df["process_canonical"] == proc) & (df["domain"].isin(domains))].copy()
        sns.boxplot(
            data=sub,
            x="domain",
            y="ai_likelihood",
            order=dom_order,
            showfliers=True,
            flierprops=flierprops(),
            ax=ax,
        )
        ax.set_title(f"{display_process(proc)} (n={len(sub)})")
        ax.set_xlabel("domain")
        ax.set_ylabel("ai_likelihood")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes_flat[len(proc_order) :]:
        ax.axis("off")

    fig.suptitle(
        f"AI likelihood by domain, faceted by process (top {len(domains)} domains; fliers are > 1.5xIQR)",
        y=1.02,
        fontsize=14,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "ai_likelihood_box_by_domain_faceted_by_process.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def build_results_dataframe_from_json(collections: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for collection in collections:
        base = ROOT / "humanization_results" / collection
        if not base.exists():
            continue
        for domain_dir in base.iterdir():
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for var_dir in domain_dir.glob("*_pangram_results"):
                if not var_dir.is_dir():
                    continue
                process = var_dir.name.replace("_pangram_results", "")
                for p in var_dir.glob("W*.json"):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    score = extract_pangram_score(data)
                    if score is None:
                        continue
                    rows.append(
                        {
                            "paper_id": data.get("paper_id") or p.stem,
                            "domain": data.get("domain") or domain,
                            "process": process,
                            "ai_likelihood": score,
                        }
                    )
    if not rows:
        raise FileNotFoundError("No fallback results could be built from humanization_results/*_pangram_results.")
    df = pd.DataFrame(rows)
    df["process"] = df["process"].astype("string")
    df["domain"] = df["domain"].astype("string")
    df["process_canonical"] = df["process"].fillna("").map(canonical_process)
    return df


def extract_pangram_score(data: dict) -> Optional[float]:
    score = data.get("ai_likelihood")
    if isinstance(score, (int, float)):
        return float(score)
    score = data.get("fraction_ai")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def load_scores_from_dir(base: Path) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {}
    if not base.exists():
        return scores

    for var_dir in base.glob("*_pangram_results"):
        if not var_dir.is_dir():
            continue
        var = canonical_process(var_dir.name.replace("_pangram_results", ""))
        vals: List[float] = []
        for p in var_dir.glob("W*.json"):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            score = extract_pangram_score(data)
            if score is not None:
                vals.append(score)
        if vals:
            scores.setdefault(var, []).extend(vals)
    return scores


def _collection_range_label(collection: str) -> str:
    return {
        "2015_back_2013": "pre-AI 2013-2015",
        "2025_back_2023": "post-AI 2023-2025",
    }.get(collection, collection)


def _load_pangram_grid_scores(collection: str) -> Dict[tuple[str, str], List[float]]:
    base = ROOT / "ai_improvement_results" / collection
    domains: tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")
    type_dirs: tuple[tuple[str, str], ...] = (
        ("original", "original"),
        ("rewritten", "polish"),
        ("improved", "refine"),
        ("new", "new"),
    )

    out: Dict[tuple[str, str], List[float]] = {}
    for dom in domains:
        dom_dir = base / dom
        if not dom_dir.exists():
            continue
        for type_dir, type_label in type_dirs:
            pdir = dom_dir / f"{type_dir}_pangram_results"
            vals: List[float] = []
            if pdir.exists():
                for p in pdir.glob("W*.json"):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    s = extract_pangram_score(data)
                    if s is not None:
                        vals.append(s)
            out[(dom, type_label)] = vals
    return out


def _load_humanized_pangram_grid_scores(collection: str) -> Dict[tuple[str, str], List[float]]:
    base = ROOT / "humanization_results" / collection
    domains: tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")
    type_dirs: tuple[tuple[str, str], ...] = (
        ("original", "original"),
        ("rewritten", "polish"),
        ("improved", "refine"),
        ("new", "new"),
    )

    out: Dict[tuple[str, str], List[float]] = {}
    for dom in domains:
        dom_dir = base / dom
        if not dom_dir.exists():
            continue
        for type_dir, type_label in type_dirs:
            pdir = dom_dir / f"{type_dir}_pangram_results"
            vals: List[float] = []
            if pdir.exists():
                for p in pdir.glob("W*.json"):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    s = extract_pangram_score(data)
                    if s is not None:
                        vals.append(s)
            out[(dom, type_label)] = vals
    return out


def render_pangram_distributions(collection: str, domain: str | None, output: Path) -> Path:
    if domain:
        raise RuntimeError("Domain filter is not supported for 4x4 pangram distribution grid.")

    domains: tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")
    type_labels: tuple[str, ...] = ("original", "polish", "refine", "new")
    scores = _load_pangram_grid_scores(collection)

    if not any(scores.values()):
        raise RuntimeError("No Pangram scores found to plot. Run pangram result generation first.")

    fig, axes = plt.subplots(
        nrows=len(domains),
        ncols=len(type_labels),
        figsize=(len(type_labels) * 3.0, len(domains) * 2.2),
        sharey=True,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    for r, dom in enumerate(domains):
        for c, typ in enumerate(type_labels):
            ax = axes[r][c]
            xs = scores.get((dom, typ), [])
            if xs:
                ax.boxplot(xs, showmeans=True)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.text(0.5, -0.20, f"n={len(xs)}", ha="center", va="top", fontsize=9, transform=ax.transAxes)
            outlier_n = count_outliers_iqr(xs)
            if outlier_n > 0:
                ax.text(
                    0.98,
                    0.95,
                    f"outliers = {outlier_n}",
                    ha="right",
                    va="top",
                    color="red",
                    fontsize=9,
                    transform=ax.transAxes,
                )
            if r == 0:
                ax.set_title(typ)
            if c == 0:
                ax.set_ylabel(dom, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"PANGRAM score distributions ({_collection_range_label(collection)})\nTypes: original / polish / refine / new",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(0.005, 0.5, "PANGRAM score", va="center", rotation="vertical")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.06, 0.03, 1, 0.93])
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def render_humanized_pangram_distributions(collection: str, domain: str | None, output: Path) -> Path:
    if domain:
        raise RuntimeError("Domain filter is not supported for 4x4 humanized pangram distribution grid.")

    domains: tuple[str, ...] = ("chemistry", "computer_science", "political_science", "theology")
    type_labels: tuple[str, ...] = ("original", "polish", "refine", "new")
    scores = _load_humanized_pangram_grid_scores(collection)

    if not any(scores.values()):
        raise RuntimeError("No humanized Pangram scores found to plot. Run humanization_pangram first.")

    fig, axes = plt.subplots(
        nrows=len(domains),
        ncols=len(type_labels),
        figsize=(len(type_labels) * 3.0, len(domains) * 2.2),
        sharey=True,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    for r, dom in enumerate(domains):
        for c, typ in enumerate(type_labels):
            ax = axes[r][c]
            xs = scores.get((dom, typ), [])
            if xs:
                ax.boxplot(xs, showmeans=True)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.text(0.5, -0.20, f"n={len(xs)}", ha="center", va="top", fontsize=9, transform=ax.transAxes)
            outlier_n = count_outliers_iqr(xs)
            if outlier_n > 0:
                ax.text(
                    0.98,
                    0.95,
                    f"outliers = {outlier_n}",
                    ha="right",
                    va="top",
                    color="red",
                    fontsize=9,
                    transform=ax.transAxes,
                )
            if r == 0:
                ax.set_title(typ)
            if c == 0:
                ax.set_ylabel(dom, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"PANGRAM score distributions after humanization ({_collection_range_label(collection)})\n"
        "Types: original / polish / refine / new",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(0.005, 0.5, "PANGRAM score", va="center", rotation="vertical")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.06, 0.03, 1, 0.93])
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def run_abstracts(args: argparse.Namespace) -> None:
    if getattr(args, "collection", None):
        collections = [args.collection]
    elif getattr(args, "collections", None):
        collections = list(args.collections)
    else:
        collections = default_collections(ROOT / "papers")
        if not collections:
            raise SystemExit(
                "No collections found under papers/ (expected papers/{collection}/metadata_{collection}.jsonl)."
            )
    for collection in collections:
        try:
            outpath = render_abstract_coverage(collection=collection, min_words=args.min_words, outdir=args.outdir)
            print(f"Wrote: {outpath}")
        except RuntimeError as exc:
            print(f"Skipped {collection}: {exc}")


def run_results(args: argparse.Namespace) -> None:
    if args.input.exists():
        outpath = render_results_boxplot(input_csv=args.input, outdir=args.outdir, max_domains=args.max_domains)
        print(f"Wrote: {outpath}")
        return

    if getattr(args, "collection", None):
        collections = [args.collection]
    elif getattr(args, "collections", None):
        collections = list(args.collections)
    else:
        collections = default_collections(ROOT / "papers")

    df = build_results_dataframe_from_json(collections)
    outpath = render_results_boxplot_from_df(df=df, outdir=args.outdir, max_domains=args.max_domains)
    print(f"Wrote: {outpath}")


def render_results_boxplot_from_df(df: pd.DataFrame, outdir: Path, max_domains: int = 16) -> Path:
    if "ai_likelihood" not in df.columns or "domain" not in df.columns or "process_canonical" not in df.columns:
        raise RuntimeError("results DataFrame is missing one of: ai_likelihood, domain, process")

    sns.set_theme(style="whitegrid")
    proc_order = process_order(df)
    if not proc_order:
        raise RuntimeError("No process values found in results data.")

    dom_counts = df["domain"].value_counts()
    domains = dom_counts.head(max_domains).index.tolist()
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
        sub = df[(df["process_canonical"] == proc) & (df["domain"].isin(domains))].copy()
        sns.boxplot(
            data=sub,
            x="domain",
            y="ai_likelihood",
            order=dom_order,
            showfliers=True,
            flierprops=flierprops(),
            ax=ax,
        )
        ax.set_title(f"{display_process(proc)} (n={len(sub)})")
        ax.set_xlabel("domain")
        ax.set_ylabel("ai_likelihood")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes_flat[len(proc_order) :]:
        ax.axis("off")

    fig.suptitle(
        f"AI likelihood by domain, faceted by process (top {len(domains)} domains; fliers are > 1.5xIQR)",
        y=1.02,
        fontsize=14,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "ai_likelihood_box_by_domain_faceted_by_process.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def run_pangram(args: argparse.Namespace) -> None:
    if getattr(args, "collection", None):
        collections = [args.collection]
    elif getattr(args, "collections", None):
        collections = list(args.collections)
    else:
        collections = default_collections(ROOT / "papers")

    for collection in collections:
        output = args.output
        if len(collections) > 1:
            output = output.parent / collection / output.name
        outpath = render_pangram_distributions(collection=collection, domain=args.domain, output=output)
        print(f"Wrote: {outpath}")


def run_humanized_pangram(args: argparse.Namespace) -> None:
    if getattr(args, "collection", None):
        collections = [args.collection]
    elif getattr(args, "collections", None):
        collections = list(args.collections)
    else:
        collections = default_collections(ROOT / "papers")

    for collection in collections:
        output = args.output
        if len(collections) > 1:
            output = output.parent / collection / output.name
        outpath = render_humanized_pangram_distributions(collection=collection, domain=args.domain, output=output)
        print(f"Wrote: {outpath}")


def run_agreement(args: argparse.Namespace) -> None:
    if getattr(args, "collection", None):
        collections = [args.collection]
    elif getattr(args, "collections", None):
        collections = list(args.collections)
    else:
        collections = default_collections(ROOT / "papers")

    for collection in collections:
        x, y, type_labels, domain_labels = load_pairs(collection)
        out = args.outdir / collection / "pangram_vs_gptzero.png"
        plot_scatter(
            collection=collection,
            x=x,
            y=y,
            type_labels=type_labels,
            domain_labels=domain_labels,
            threshold=args.agreement_threshold,
            out_path=out,
        )
        print(f"Wrote: {out} (n={len(x)})")


def run_all(args: argparse.Namespace) -> None:
    abstracts_args = argparse.Namespace(
        collection=getattr(args, "collection", None),
        collections=getattr(args, "collections", None),
        min_words=args.min_words,
        outdir=args.outdir,
    )
    run_abstracts(abstracts_args)

    results_args = argparse.Namespace(
        input=args.input,
        outdir=args.outdir,
        max_domains=args.max_domains,
        collection=getattr(args, "collection", None),
        collections=getattr(args, "collections", None),
    )
    try:
        run_results(results_args)
    except FileNotFoundError:
        print(f"Skipped results plot: input not found at {results_args.input}")

    pangram_args = argparse.Namespace(
        collection=getattr(args, "collection", None),
        collections=getattr(args, "collections", None),
        domain=args.domain,
        output=args.pangram_output,
    )
    try:
        run_pangram(pangram_args)
    except RuntimeError as exc:
        print(f"Skipped pangram distributions: {exc}")

    humanized_pangram_args = argparse.Namespace(
        collection=getattr(args, "collection", None),
        collections=getattr(args, "collections", None),
        domain=args.domain,
        output=args.humanized_pangram_output,
    )
    try:
        run_humanized_pangram(humanized_pangram_args)
    except RuntimeError as exc:
        print(f"Skipped humanized pangram distributions: {exc}")

    agreement_args = argparse.Namespace(
        collection=getattr(args, "collection", None),
        collections=getattr(args, "collections", None),
        outdir=args.outdir,
        agreement_threshold=args.agreement_threshold,
    )
    run_agreement(agreement_args)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate project figures from one script.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_abstracts = sub.add_parser("abstracts", help="Render abstract coverage figure(s).")
    p_abstracts.add_argument("--collection", default=None)
    p_abstracts.add_argument("--collections", nargs="*", default=None)
    p_abstracts.add_argument("--min-words", type=int, default=25)
    p_abstracts.add_argument("--outdir", type=Path, default=ROOT / "results" / "figures")
    p_abstracts.set_defaults(func=run_abstracts)

    p_results = sub.add_parser("results", help="Render results.csv process/domain boxplot.")
    p_results.add_argument("--input", type=Path, default=ROOT / "results" / "results.csv")
    p_results.add_argument("--outdir", type=Path, default=ROOT / "results" / "figures")
    p_results.add_argument("--max-domains", type=int, default=16)
    p_results.set_defaults(func=run_results)

    p_pangram = sub.add_parser("pangram", help="Render pangram distribution boxplot.")
    p_pangram.add_argument("--collection", default=None)
    p_pangram.add_argument("--collections", nargs="*", default=None)
    p_pangram.add_argument("--domain", default=None)
    p_pangram.add_argument("--output", type=Path, default=ROOT / "results" / "figures" / "pangram_distributions.png")
    p_pangram.set_defaults(func=run_pangram)

    p_humanized_pangram = sub.add_parser(
        "pangram-humanized",
        help="Render 4x4 pangram distribution grid after humanization.",
    )
    p_humanized_pangram.add_argument("--collection", default=None)
    p_humanized_pangram.add_argument("--collections", nargs="*", default=None)
    p_humanized_pangram.add_argument("--domain", default=None)
    p_humanized_pangram.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "figures" / "pangram_distributions_humanized.png",
    )
    p_humanized_pangram.set_defaults(func=run_humanized_pangram)

    p_agreement = sub.add_parser("agreement", help="Render Pangram vs GPTZero agreement scatter figure(s).")
    p_agreement.add_argument("--collection", default=None)
    p_agreement.add_argument("--collections", nargs="*", default=None)
    p_agreement.add_argument("--outdir", type=Path, default=ROOT / "results" / "figures")
    p_agreement.add_argument("--agreement-threshold", type=float, default=0.5)
    p_agreement.set_defaults(func=run_agreement)


    p_all = sub.add_parser("all", help="Run all figure generators.")
    p_all.add_argument("--collection", default=None)
    p_all.add_argument("--collections", nargs="*", default=None)
    p_all.add_argument("--min-words", type=int, default=25)
    p_all.add_argument("--outdir", type=Path, default=ROOT / "results" / "figures")
    p_all.add_argument("--input", type=Path, default=ROOT / "results" / "results.csv")
    p_all.add_argument("--max-domains", type=int, default=16)
    p_all.add_argument("--domain", default=None)
    p_all.add_argument("--pangram-output", type=Path, default=ROOT / "results" / "figures" / "pangram_distributions.png")
    p_all.add_argument(
        "--humanized-pangram-output",
        type=Path,
        default=ROOT / "results" / "figures" / "pangram_distributions_humanized.png",
    )
    p_all.add_argument("--agreement-threshold", type=float, default=0.5)
    p_all.set_defaults(func=run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
