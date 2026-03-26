#!/usr/bin/env python3
"""
Run detector scoring on Undetectable-humanized abstracts.

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
  humanization_results/{collection}/{domain}/{original|improved|new|rewritten}_{detector}_results/{paper_id}.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from pangram import Pangram
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

HUMANIZATION_RESULTS_DIR = ROOT / "humanization_results"
HUMANIZATION_DIR = ROOT / "humanization"

DEFAULT_COLLECTION = "2025_back_2023"
DOMAINS = ["chemistry", "computer_science", "political_science", "theology"]
VARIANTS = ["original", "improved", "new", "rewritten"]

load_dotenv()
PANGRAM_API_KEY = os.getenv("PANGRAM_API")
GPT_ZERO_API_KEY = os.getenv("GPT_ZERO_API_KEY") or os.getenv("GPTZERO_API_KEY")
GPTZERO_ENDPOINT = "https://api.gptzero.me/v2/predict/text"


def _iter_domains(domains: Optional[Iterable[str]]) -> List[str]:
    return DOMAINS if domains is None else list(domains)


def _iter_variants(variants: Optional[Iterable[str]]) -> List[str]:
    return VARIANTS if variants is None else list(variants)


def get_pangram_result(text: str, client: Pangram) -> Optional[dict]:
    if not text or not text.strip():
        return None
    try:
        return client.predict(text)
    except Exception as e:
        print(f"\n❌ PANGRAM error: {type(e).__name__}: {e}")
        return None


def get_gptzero_result(text: str, api_key: str) -> Optional[dict]:
    if not text or not text.strip():
        return None
    payload = {"document": text.strip(), "multilingual": False}
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(GPTZERO_ENDPOINT, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents") or []
        if not docs:
            return None
        doc = docs[0]
        result = dict(doc.get("class_probabilities") or {})
        if doc.get("document_classification") is not None:
            result["document_class"] = doc.get("document_classification")
        return result
    except Exception as e:
        print(f"\n❌ GPTZero error: {type(e).__name__}: {e}")
        return None


def build_input_items(
    collection: str,
    domains: Optional[Iterable[str]],
    variants: Optional[Iterable[str]],
) -> List[Tuple[str, str, Path]]:
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


def process_humanization_with_detector(
    collection: str,
    detector: str,
    domains: Optional[Iterable[str]] = None,
    variants: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    client: Optional[Pangram] = None
    if detector == "pangram":
        if not PANGRAM_API_KEY:
            print("Error: PANGRAM_API key not found in .env file")
            return
        client = Pangram(api_key=PANGRAM_API_KEY)
        print("✅ PANGRAM client initialized\n")
    elif detector == "gptzero":
        if not GPT_ZERO_API_KEY:
            print("Error: GPT_ZERO_API_KEY not found in .env file")
            return
        print("✅ GPTZero API key found\n")
    else:
        print(f"Error: Unsupported detector '{detector}'")
        return

    items = build_input_items(collection, domains, variants)
    print(f"Found {len(items)} humanization abstracts to consider.")

    processed = written = skipped = failed = 0
    desc = "PANGRAM humanization" if detector == "pangram" else "GPTZero humanization"
    for dom, var, in_path in tqdm(items, desc=desc, unit="abstract"):
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
            failed += 1
            continue

        out_dir = HUMANIZATION_RESULTS_DIR / collection / domain / f"{var}_{detector}_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{paper_id}.json"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        if detector == "pangram":
            assert client is not None
            detector_result = get_pangram_result(humanized_abstract, client)
        else:
            detector_result = get_gptzero_result(humanized_abstract, GPT_ZERO_API_KEY or "")
        if not detector_result:
            failed += 1
            continue

        result_entry = {
            "paper_id": paper_id,
            "domain": domain,
            "variant": var,
            "text": humanized_abstract,
            **detector_result,
        }
        try:
            with out_path.open("w") as f:
                json.dump(result_entry, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            print(f"\n⚠️  Failed to write {out_path}: {e}")
            failed += 1

    print("\nSummary:")
    print(f"  Total input abstracts seen: {len(items)}")
    print(f"  Processed (subject to limit): {processed}")
    print(f"  Written new results: {written}")
    print(f"  Skipped existing: {skipped}")
    print(f"  Failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run detector scoring on Undetectable-humanized abstracts and save per-paper results\n"
            "under humanization_results/{collection}/{domain}/{variant}_{detector}_results/."
        )
    )
    parser.add_argument("--detector", default="pangram", choices=["pangram", "gptzero"])
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--domains", nargs="*", choices=DOMAINS)
    parser.add_argument("--variants", nargs="*", choices=VARIANTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_humanization_with_detector(
        collection=args.collection,
        detector=args.detector,
        domains=args.domains,
        variants=args.variants,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
