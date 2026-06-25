#!/usr/bin/env python3
"""
Run detector scoring on Undetectable-humanized abstracts, in parallel.

This is a multithreaded variant of ``humanization_ai_detection.py``. Detection
requests for individual abstracts are dispatched across a thread pool (8 workers
by default, configurable via ``--workers``).

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
  humanization_results/{collection}/{domain}/{variant}_{detector}_results/{paper_id}.json
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pangram import Pangram
from pydantic import BaseModel, Field
from tqdm import tqdm
from variants import VARIANTS, normalize_variant_raw, result_dir_name

ROOT = Path(__file__).resolve().parents[1]

HUMANIZATION_RESULTS_DIR = ROOT / "humanization_results"
HUMANIZATION_DIR = ROOT / "humanization"

DEFAULT_COLLECTION = "2025_back_2023"
DEFAULT_WORKERS = 8
DOMAINS = ["chemistry", "computer_science", "political_science", "theology"]
def _read_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


# Prefer project .env values over inherited shell env values.
load_dotenv(override=True)
PANGRAM_API_KEY = _read_env("PANGRAM_API")
GPT_ZERO_API_KEY = _read_env("GPT_ZERO_API_KEY") or _read_env("GPTZERO_API_KEY")
GPTZERO_ENDPOINT = "https://api.gptzero.me/v2/predict/text"
OPENROUTER_API_KEY = _read_env("OPENROUTER_API_KEY")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
LLM_ASSISTED_MODEL_NAME = "openai/gpt-5-nano"


class LLMAiDetectionResult(BaseModel):
    result_explanation: str = Field(
        description="Explanation of why this probability was chosen. 1-2 short sentences",
    )
    ai_probability: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
            le=100,
            description="Probability that the input text is AI-generated, percentage (0-100)",
        ),
    ]


LLM_AI_DETECTION_PROMPT = (
    """
You are an expert scientist who excels in detecting AI in texts. Your task is to check whether the text given to you in the user's message is AI-generated or not.
Provide a probability (whole number, 0-100) and a short explanation.
Respond as JSON only, following this schema:
""".strip()
    + "\n"
    + json.dumps(LLMAiDetectionResult.model_json_schema())
)


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


def get_llm_assisted_result(text: str, client: OpenAI) -> Optional[dict]:
    if not text or not text.strip():
        return None
    try:
        response = client.chat.completions.create(
            model=LLM_ASSISTED_MODEL_NAME,
            messages=[
                {"role": "system", "content": LLM_AI_DETECTION_PROMPT},
                {"role": "user", "content": text.strip()},
            ],
            response_format={"type": "json_object"},
        )
        if not response.choices:
            return None
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = LLMAiDetectionResult.model_validate(json.loads(content))
        return {
            "ai_probability": parsed.ai_probability / 100,
            "explanation": parsed.result_explanation,
            "model_name": LLM_ASSISTED_MODEL_NAME,
        }
    except Exception as e:
        print(f"\n❌ LLM-assisted detection error: {type(e).__name__}: {e}")
        return None


def check_llm_assisted_auth(client: OpenAI) -> bool:
    """
    Make a tiny probe request so we fail fast on invalid OpenRouter auth.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_ASSISTED_MODEL_NAME,
            messages=[{"role": "user", "content": "Return a JSON object with ai_probability=50 and result_explanation='test'."}],
            response_format={"type": "json_object"},
            max_tokens=64,
        )
        return bool(response.choices)
    except Exception as e:
        print(
            "\n❌ LLM-assisted auth probe failed.\n"
            f"   Error: {type(e).__name__}: {e}\n"
            "   OpenRouter is rejecting this key for chat completions.\n"
            "   Please verify OPENROUTER_API_KEY has active credits and valid generation access."
        )
        return False


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


def _process_one_item(
    item: Tuple[str, str, Path],
    collection: str,
    detector: str,
    overwrite: bool,
    client: Optional[Pangram],
    llm_client: Optional[OpenAI],
) -> str:
    """
    Process a single abstract and return one of:
      "written", "skipped", "failed".
    Safe to run concurrently from a thread pool.
    """
    dom, var, in_path = item
    try:
        with in_path.open() as f:
            src = json.load(f)
    except Exception as e:
        print(f"\n⚠️  Failed to read {in_path}: {e}")
        return "failed"

    paper_id = src.get("paper_id") or in_path.stem
    domain = src.get("domain") or dom
    humanized_abstract = src.get("humanized_abstract")
    if not humanized_abstract:
        return "failed"

    var = normalize_variant_raw(var)
    out_dir = HUMANIZATION_RESULTS_DIR / collection / domain / result_dir_name(var, detector)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper_id}.json"
    if out_path.exists() and not overwrite:
        return "skipped"

    if detector == "pangram":
        assert client is not None
        detector_result = get_pangram_result(humanized_abstract, client)
    elif detector == "gptzero":
        detector_result = get_gptzero_result(humanized_abstract, GPT_ZERO_API_KEY or "")
    else:
        assert llm_client is not None
        detector_result = get_llm_assisted_result(humanized_abstract, llm_client)
    if not detector_result:
        return "failed"

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
        return "written"
    except Exception as e:
        print(f"\n⚠️  Failed to write {out_path}: {e}")
        return "failed"


def process_humanization_with_detector(
    collection: str,
    detector: str,
    domains: Optional[Iterable[str]] = None,
    variants: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> None:
    client: Optional[Pangram] = None
    llm_client: Optional[OpenAI] = None
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
    elif detector == "llm_assisted":
        if not OPENROUTER_API_KEY:
            print("Error: OPENROUTER_API_KEY not found in .env file")
            return
        llm_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_ENDPOINT,
        )
        if not check_llm_assisted_auth(llm_client):
            return
        print(f"✅ LLM-assisted detector initialized ({LLM_ASSISTED_MODEL_NAME})\n")
    else:
        print(f"Error: Unsupported detector '{detector}'")
        return

    items = build_input_items(collection, domains, variants)
    if limit is not None:
        items = items[:limit]
    print(f"Found {len(items)} humanization abstracts to consider.")

    workers = max(1, workers)
    print(f"Running detection with {workers} worker thread(s).\n")

    written = skipped = failed = 0
    counts_lock = threading.Lock()

    if detector == "pangram":
        desc = "PANGRAM humanization"
    elif detector == "gptzero":
        desc = "GPTZero humanization"
    else:
        desc = "LLM-assisted humanization"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_one_item,
                item,
                collection,
                detector,
                overwrite,
                client,
                llm_client,
            )
            for item in items
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=desc, unit="abstract"
        ):
            try:
                status = future.result()
            except Exception as e:
                print(f"\n⚠️  Unexpected worker error: {type(e).__name__}: {e}")
                status = "failed"
            with counts_lock:
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1

    print("\nSummary:")
    print(f"  Total input abstracts seen: {len(items)}")
    print(f"  Processed: {len(items)}")
    print(f"  Written new results: {written}")
    print(f"  Skipped existing: {skipped}")
    print(f"  Failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run detector scoring on Undetectable-humanized abstracts in parallel and save\n"
            "per-paper results under humanization_results/{collection}/{domain}/"
            "{variant}_{detector}_results/."
        )
    )
    parser.add_argument(
        "--detector",
        default="pangram",
        choices=["pangram", "gptzero", "llm_assisted"],
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--domains", nargs="*", choices=DOMAINS)
    parser.add_argument("--variants", nargs="*", choices=VARIANTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel worker threads (default: {DEFAULT_WORKERS})",
    )
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
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
