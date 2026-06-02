"""Canonical variant and detector names for on-disk layout and analysis."""

from __future__ import annotations

# On-disk variant folders and variant_raw in JSON
VARIANTS = (
    "original",
    "refine_abstract_only",
    "refine_abstract_article",
    "new_article_only",
)

VARIANT_LABEL = {
    "original": "original",
    "refine_abstract_only": "refine (abstract only)",
    "refine_abstract_article": "refine (abstract + article)",
    "new_article_only": "new (article only)",
}

# ai_improvement/{collection}/{domain}/{subdir}/
AI_IMPROVEMENT_SUBDIRS = {
    "refine_abstract_only": "refine_abstract_only",
    "refine_abstract_article": "refine_abstract_article",
    "new_article_only": "new_article_only",
}

LEGACY_VARIANT_RAW = {
    "rewritten": "refine_abstract_only",
    "polish": "refine_abstract_only",
    "rewrite": "refine_abstract_only",
    "improved": "refine_abstract_article",
    "refine": "refine_abstract_article",
    "improve": "refine_abstract_article",
    "new": "new_article_only",
    "original": "original",
}

LEGACY_AI_IMPROVEMENT_DIR = {
    "rewritten_abstracts": "refine_abstract_only",
    "improved_abstracts": "refine_abstract_article",
    "new_abstracts": "new_article_only",
}

DETECTORS = ("pangram", "gptzero", "llm_assisted")

LEGACY_DETECTOR = {
    "llm_aid": "llm_assisted",
}

RESULT_DIR_DETECTORS = ("llm_assisted", "llm_aid", "gptzero", "pangram")


def normalize_variant_raw(value: str) -> str:
    v = (value or "").strip().lower()
    if v in VARIANT_LABEL:
        return v
    if v in LEGACY_VARIANT_RAW:
        return LEGACY_VARIANT_RAW[v]
    if v in {"refine (abstract only)"}:
        return "refine_abstract_only"
    if v in {"refine (abstract + paper)", "refine (abstract + article)"}:
        return "refine_abstract_article"
    if v in {"new (article only)"}:
        return "new_article_only"
    return v


def normalize_detector(value: str) -> str:
    d = (value or "").strip().lower()
    return LEGACY_DETECTOR.get(d, d)


def result_dir_name(variant_raw: str, detector: str) -> str:
    v = normalize_variant_raw(variant_raw)
    d = normalize_detector(detector)
    return f"{v}_{d}_results"


def parse_result_dir(dirname: str) -> tuple[str, str] | None:
    if not dirname.endswith("_results"):
        return None
    stem = dirname[: -len("_results")]
    for det in RESULT_DIR_DETECTORS:
        suffix = f"_{det}"
        if stem.endswith(suffix):
            variant = stem[: -len(suffix)]
            return variant, det
    return None


def canonical_result_dir(dirname: str) -> str | None:
    parsed = parse_result_dir(dirname)
    if parsed is None:
        return None
    variant, detector = parsed
    new_variant = normalize_variant_raw(variant)
    new_detector = normalize_detector(detector)
    if variant == new_variant and detector == new_detector:
        return None
    return result_dir_name(new_variant, new_detector)
