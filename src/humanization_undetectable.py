#!/usr/bin/env python3
"""
Humanize abstracts with Undetectable.AI and save rich results.

Inputs (discovered from existing project structure; no prior minimal humanization JSON required):

- Original abstracts:
    papers/{collection}/{domain}/paper_jsons/{paper_id}.json
      {
        "id": "...",
        "domain": "...",
        "abstract": "..."
        ...
      }

- Rewrite abstracts:
    ai_improvement/{collection}/{domain}/refine_abstract_only/{paper_id}.json
    ai_improvement/{collection}/{domain}/refine_abstract_article/{paper_id}.json
    ai_improvement/{collection}/{domain}/new_article_only/{paper_id}.json
      {
        "id": "...",
        "domain": "...",
        "abstract": "..."
      }

Outputs (one combined JSON per paper/variant, under humanization/):

  humanization/{collection}/{domain}/{variant}/{paper_id}.json
    {
      "paper_id": "...",
      "domain": "...",
      "variant": "original" | "refine_abstract_only" | "refine_abstract_article" | "new_article_only",
      "original_abstract": "...",
      "humanized_abstract": "...",
      "undetectable": {
        "params": { ... params we sent ... },
        "document": { ... full /document response from Undetectable ... }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from variants import AI_IMPROVEMENT_SUBDIRS, VARIANTS


ROOT = Path(__file__).resolve().parents[1]

# Undetectable outputs live under humanization/; downstream model runs (pangram, gptzero) live under humanization_results/.
HUMANIZATION_DIR = ROOT / "humanization"

DEFAULT_COLLECTION = "2025_back_2023"
DOMAINS = ["chemistry", "computer_science", "political_science", "theology"]
UNDETECTABLE_SUBMIT_URL = "https://humanize.undetectable.ai/submit"
UNDETECTABLE_DOCUMENT_URL = "https://humanize.undetectable.ai/document"
UNDETECTABLE_CREDITS_URL = "https://humanize.undetectable.ai/check-user-credits"

# Load environment variables from .env file
load_dotenv()
UNDETECTABLE_API_KEY = os.getenv("UNDETECTABLE_API_KEY")


def _iter_domains(domains: Optional[Iterable[str]]) -> List[str]:
    if domains is None:
        return DOMAINS
    return list(domains)


def _iter_variants(variants: Optional[Iterable[str]]) -> List[str]:
    if variants is None:
        return VARIANTS
    return list(variants)


def _count_words(text: str) -> int:
    if not text:
        return 0
    # Simple whitespace-based word count is sufficient for estimating credit use.
    return len(text.split())


def undetectable_check_credits() -> Optional[int]:
    """
    Query Undetectable for current word credits.

    Returns available credits (int) or None if the check fails.
    """
    headers = {
        "apikey": UNDETECTABLE_API_KEY or "",
        "accept": "application/json",
    }
    try:
        resp = requests.get(UNDETECTABLE_CREDITS_URL, headers=headers, timeout=30)
    except Exception as e:
        print(f"\n⚠️  Failed to check Undetectable credits: {type(e).__name__}: {e}")
        return None

    if resp.status_code != 200:
        print(f"\n⚠️  Failed to check Undetectable credits ({resp.status_code}): {resp.text[:300]}")
        return None

    try:
        data = resp.json()
    except Exception as e:
        print(f"\n⚠️  Could not parse Undetectable credits response: {e}")
        return None

    credits = data.get("credits")
    if isinstance(credits, int):
        return credits
    if isinstance(credits, float):
        return int(credits)
    return None


def build_input_items(
    collection: str,
    domains: Optional[Iterable[str]],
    variants: Optional[Iterable[str]],
) -> List[Tuple[str, str, Path]]:
    """
    Collect all (domain, variant, input_path) triples to process directly from the existing structure:

    - original:  papers/{collection}/{domain}/paper_jsons/W*.json
    - refine_* / new_article_only: ai_improvement/{collection}/{domain}/{variant}/W*.json
    """
    items: List[Tuple[str, str, Path]] = []

    papers_base = ROOT / "papers" / collection
    ai_imp_base = ROOT / "ai_improvement" / collection
    selected = set(_iter_variants(variants))

    for dom in _iter_domains(domains):
        if "original" in selected:
            paper_json_dir = papers_base / dom / "paper_jsons"
            if paper_json_dir.exists():
                for path in sorted(paper_json_dir.glob("W*.json")):
                    if path.is_file():
                        items.append((dom, "original", path))

        for variant in AI_IMPROVEMENT_SUBDIRS:
            if variant not in selected:
                continue
            var_dir = ai_imp_base / dom / variant
            if var_dir.exists():
                for path in sorted(var_dir.glob("W*.json")):
                    if path.is_file():
                        items.append((dom, variant, path))

    return items


def undetectable_submit_and_wait(
    text: str,
    readability: str = "Doctorate",
    purpose: str = "Article",
    strength: str = "Balanced",
    model: str = "v11",
    max_polls: int = 30,
    poll_interval_sec: float = 2.0,
) -> Optional[dict]:
    """
    Submit text to Undetectable /submit, then poll /document until output is ready.

    Returns the final /document JSON (with input/output/readability/purpose/createdDate, etc.)
    or None on failure.
    """
    headers = {
        "apikey": UNDETECTABLE_API_KEY or "",
        "Content-Type": "application/json",
    }
    payload = {
        "content": text,
        "readability": readability,
        "purpose": purpose,
        "strength": strength,
        "model": model,
    }

    try:
        submit_resp = requests.post(UNDETECTABLE_SUBMIT_URL, headers=headers, json=payload, timeout=60)
    except Exception as e:
        print(f"\n❌ Undetectable submit error: {type(e).__name__}: {e}")
        return None

    if submit_resp.status_code != 200:
        print(f"\n❌ Undetectable submit failed ({submit_resp.status_code}): {submit_resp.text[:500]}")
        return None

    try:
        submit_data = submit_resp.json()
    except Exception as e:
        print(f"\n❌ Could not parse submit response JSON: {e}")
        return None

    doc_id = submit_data.get("id")
    if not doc_id:
        print(f"\n❌ Undetectable submit response missing 'id': {submit_data}")
        return None

    # Poll /document until output is present or we hit max_polls
    doc_payload = {"id": doc_id}
    for _ in range(max_polls):
        try:
            doc_resp = requests.post(UNDETECTABLE_DOCUMENT_URL, headers=headers, json=doc_payload, timeout=60)
        except Exception as e:
            print(f"\n❌ Undetectable document error: {type(e).__name__}: {e}")
            return None

        if doc_resp.status_code != 200:
            # For not-yet-ready documents, API may still return 200; treat non-200 as fatal.
            print(f"\n❌ Undetectable document failed ({doc_resp.status_code}): {doc_resp.text[:500]}")
            return None

        try:
            doc_data = doc_resp.json()
        except Exception as e:
            print(f"\n❌ Could not parse document response JSON: {e}")
            return None

        # According to docs, 'output' is the humanized text, 'input' is original.
        if doc_data.get("output"):
            return doc_data

        # Not done yet; wait and poll again.
        time.sleep(poll_interval_sec)

    print(f"\n⚠️ Undetectable document never became ready for id={doc_id}")
    return None


def process_humanization_with_undetectable(
    collection: str,
    domains: Optional[Iterable[str]] = None,
    variants: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    readability: str = "Doctorate",
    purpose: str = "Article",
    strength: str = "Balanced",
    model: str = "v11",
) -> None:
    """
    Call Undetectable on abstracts discovered from papers/ and ai_improvement/ and write
    combined results under humanization/{collection}/{domain}/{variant}_undetectable_results/.
    """
    if not UNDETECTABLE_API_KEY:
        print("Error: UNDETECTABLE_API_KEY not set in .env")
        return

    items = build_input_items(collection, domains, variants)
    total_inputs = len(items)
    print(f"Found {total_inputs} abstracts to consider for Undetectable humanization.")

    # Pre-flight: estimate total word usage and compare to available credits, if possible.
    if total_inputs > 0:
        estimated_words = 0
        for _, _, in_path in items:
            try:
                with in_path.open() as f:
                    src = json.load(f)
            except Exception:
                continue
            text = src.get("abstract") or src.get("text") or ""
            estimated_words += _count_words(text)

        print(f"Estimated total words to humanize: {estimated_words}")
        credits = undetectable_check_credits()
        if credits is not None:
            print(f"Undetectable credits available: {credits}")
            if credits < estimated_words:
                shortfall = estimated_words - credits
                print(
                    "\n❌ Not enough Undetectable credits to humanize all requested texts.\n"
                    f"   Needed ≈ {estimated_words} words, have {credits} (short by ≈ {shortfall}).\n"
                    "   No requests were sent; reduce scope (domains/variants/limit) or add credits."
                )
                return

    processed = 0
    written = 0
    skipped = 0
    failed = 0

    for dom, var, in_path in tqdm(items, desc="Undetectable humanization", unit="abstract"):
        if limit is not None and processed >= limit:
            break

        processed += 1

        try:
            with in_path.open() as f:
                src = json.load(f)
        except Exception as e:
            print(f"\n⚠️ Failed to read {in_path}: {e}")
            failed += 1
            continue

        # All source JSONs (paper_jsons and ai_improvement *_abstracts) have id/domain/abstract.
        paper_id = src.get("id") or src.get("paper_id") or in_path.stem
        domain = src.get("domain") or dom
        original_abstract = src.get("abstract")

        if not original_abstract or len(original_abstract.strip()) < 50:
            # Undetectable requires at least 50 characters.
            failed += 1
            continue

        out_dir = HUMANIZATION_DIR / collection / domain / var
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{paper_id}.json"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        doc_data = undetectable_submit_and_wait(
            original_abstract,
            readability=readability,
            purpose=purpose,
            strength=strength,
            model=model,
        )
        if not doc_data:
            failed += 1
            continue

        humanized = doc_data.get("output")
        if not humanized:
            failed += 1
            continue

        result_entry = {
            "paper_id": paper_id,
            "domain": domain,
            "humanizer": "undetectable",
            "variant": var,
            "original_abstract": original_abstract,
            "humanized_abstract": humanized,
            "undetectable": {
                "params": {
                    "readability": readability,
                    "purpose": purpose,
                    "strength": strength,
                    "model": model,
                },
                "document": doc_data,
            },
        }

        try:
            with out_path.open("w") as f:
                json.dump(result_entry, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            print(f"\n⚠️ Failed to write {out_path}: {e}")
            failed += 1

    print("\nSummary:")
    print(f"  Total input abstracts seen: {total_inputs}")
    print(f"  Processed (subject to limit): {processed}")
    print(f"  Written new results: {written}")
    print(f"  Skipped existing: {skipped}")
    print(f"  Failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Undetectable.AI humanization on abstracts discovered from papers/ and ai_improvement,\n"
            "and save original + humanized text plus full Undetectable /document data under\n"
            "humanization/{collection}/{domain}/{variant}_undetectable_results/."
        )
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection folder under humanization/ (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        choices=DOMAINS,
        help="Optional list of domains to restrict processing to.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=VARIANTS,
        help="Optional list of text variants to process (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of input abstracts to process (for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON result files if they already exist.",
    )
    parser.add_argument(
        "--readability",
        default="Doctorate",
        choices=["High School", "University", "Doctorate", "Journalist", "Marketing"],
        help="Undetectable readability setting (default: Doctorate).",
    )
    parser.add_argument(
        "--purpose",
        default="Article",
        choices=[
            "General Writing",
            "Essay",
            "Article",
            "Marketing Material",
            "Story",
            "Cover Letter",
            "Report",
            "Business Material",
            "Legal Material",
        ],
        help="Undetectable purpose setting (default: Article).",
    )
    parser.add_argument(
        "--strength",
        default="Balanced",
        choices=["Quality", "Balanced", "More Human"],
        help="Undetectable strength (aggressiveness) setting (default: Balanced).",
    )
    parser.add_argument(
        "--model",
        default="v11",
        choices=["v2", "v11", "v11sr"],
        help="Undetectable model version (default: v11).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_humanization_with_undetectable(
        collection=args.collection,
        domains=args.domains,
        variants=args.variants,
        limit=args.limit,
        overwrite=args.overwrite,
        readability=args.readability,
        purpose=args.purpose,
        strength=args.strength,
        model=args.model,
    )


if __name__ == "__main__":
    main()

