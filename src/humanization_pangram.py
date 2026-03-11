#!/usr/bin/env python3
"""
Run PANGRAM AI detection on Undetectable-humanized abstracts.

Inputs (per paper_id/domain, already split by variant):
  humanization/{collection}/{domain}/{variant}/{paper_id}.json
    {
      "paper_id": "...",
      "domain": "...",
      "original_abstract": "...",
      "humanized_abstract": "...",
      "undetectable": { ... }
    }

Outputs (same folder structure as ai_improvement_results, but under humanization_results/):
  humanization_results/{collection}/{domain}/{original|improved|new|rewritten}_pangram_results/{paper_id}.json
    {
      "paper_id": "...",
      "domain": "...",
      "variant": "original" | "improved" | "new" | "rewritten",
      "text": "<the humanized abstract sent to Pangram>",
      ... all PANGRAM response fields ...
    }
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from pangram import Pangram


ROOT = Path(__file__).resolve().parents[1]

HUMANIZATION_RESULTS_DIR = ROOT / "humanization_results"
HUMANIZATION_DIR = ROOT / "humanization"

DEFAULT_COLLECTION = "2025_back_2023"
DOMAINS = ["chemistry", "computer_science", "political_science", "theology"]
VARIANTS = ["original", "improved", "new", "rewritten"]

# Load environment variables from .env file
load_dotenv()
PANGRAM_API_KEY = os.getenv("PANGRAM_API")


def _iter_domains(domains: Optional[Iterable[str]]) -> List[str]:
    if domains is None:
        return DOMAINS
    return list(domains)


def _iter_variants(variants: Optional[Iterable[str]]) -> List[str]:
    if variants is None:
        return VARIANTS
    return list(variants)


def get_pangram_result(text: str, client: Pangram) -> Optional[dict]:
    """
    Get AI detection result from PANGRAM API using the SDK.
    Returns the full API response as a dictionary, or None if failed.
    """
    if not text or not text.strip():
        return None
    try:
        return client.predict(text)
    except Exception as e:
        print(f"\n❌ PANGRAM error: {type(e).__name__}: {e}")
        return None


def build_input_items(
    collection: str,
    domains: Optional[Iterable[str]],
    variants: Optional[Iterable[str]],
) -> List[Tuple[str, str, Path]]:
    """
    Collect all (domain, variant, input_path) triples to process.

    Each input_path points to a Undetectable result JSON with field 'humanized_abstract'.
    """
    items: List[Tuple[str, str, Path]] = []
    for dom in _iter_domains(domains):
        for var in _iter_variants(variants):
            in_dir = HUMANIZATION_DIR / collection / dom / var
            if not in_dir.exists():
                continue
            for path in sorted(in_dir.glob("W*.json")):
                if path.is_file():
                    items.append((dom, var, path))
    return items


def process_humanization_with_pangram(
    collection: str,
    domains: Optional[Iterable[str]] = None,
    variants: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """
    Run PANGRAM on humanization abstracts and write results into humanization_results tree.
    """
    if not PANGRAM_API_KEY:
        print("Error: PANGRAM_API key not found in .env file")
        return

    try:
        client = Pangram(api_key=PANGRAM_API_KEY)
        print("✅ PANGRAM client initialized\n")
    except Exception as e:
        print(f"Error initializing PANGRAM client: {e}")
        return

    items = build_input_items(collection, domains, variants)
    total_inputs = len(items)
    print(f"Found {total_inputs} humanization abstracts to consider.")

    processed = 0
    written = 0
    skipped = 0
    failed = 0

    for dom, var, in_path in tqdm(items, desc="PANGRAM humanization", unit="abstract"):
        if limit is not None and processed >= limit:
            break

        processed += 1

        try:
            with in_path.open() as f:
                src = json.load(f)
        except Exception as e:
            print(f"\n⚠️  Failed to read {in_path}: {e}")
            failed += 1
            continue

        paper_id = src.get("paper_id") or in_path.stem
        domain = src.get("domain") or dom
        humanized_abstract = src.get("humanized_abstract")

        if not humanized_abstract:
            # Nothing to send to Pangram
            failed += 1
            continue

        # Map variant -> output subfolder name
        out_subdir_name = f"{var}_pangram_results"
        out_dir = HUMANIZATION_RESULTS_DIR / collection / domain / out_subdir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{paper_id}.json"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        pangram_result = get_pangram_result(humanized_abstract, client)
        if not pangram_result:
            failed += 1
            continue

        result_entry = {
            "paper_id": paper_id,
            "domain": domain,
            "variant": var,
            "text": humanized_abstract,
            **pangram_result,
        }

        try:
            with out_path.open("w") as f:
                json.dump(result_entry, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            print(f"\n⚠️  Failed to write {out_path}: {e}")
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
            "Run PANGRAM AI detection on Undetectable-humanized abstracts and save per-paper results\n"
            "under humanization_results/{collection}/{domain}/{variant}_pangram_results/."
        )
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection folder (default: {DEFAULT_COLLECTION})",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_humanization_with_pangram(
        collection=args.collection,
        domains=args.domains,
        variants=args.variants,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

