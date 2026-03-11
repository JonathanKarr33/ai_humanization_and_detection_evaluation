from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]


def _domains_iter(domains: Optional[Iterable[str]]) -> Iterable[str]:
    if domains is None:
        return ["chemistry", "computer_science", "political_science", "theology"]
    return list(domains)


def run(
    collection: str,
    domains: Optional[List[str]] = None,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    Main entry point.

    - Reads originals from papers/{collection}/{domain}/paper_jsons/W*.json
    - Reads improved/new/rewritten abstracts from ai_improvement/{collection}/{domain}/*_abstracts/W*.json
    - Writes JSONs into:
        humanization/{collection}/{domain}/original/{paper_id}.json
        humanization/{collection}/{domain}/improved/{paper_id}.json
        humanization/{collection}/{domain}/new/{paper_id}.json
        humanization/{collection}/{domain}/rewritten/{paper_id}.json
    - Skips any output file that already exists unless --overwrite is set
    """
    base_papers = ROOT / "papers" / collection
    base_ai_imp = ROOT / "ai_improvement" / collection

    total_papers = 0
    processed_papers = 0
    written_original = 0
    written_improved = 0
    written_new = 0
    written_rewritten = 0
    skipped_original = 0
    skipped_improved = 0
    skipped_new = 0
    skipped_rewritten = 0

    for dom in _domains_iter(domains):
        paper_json_dir = base_papers / dom / "paper_jsons"
        improved_dir = base_ai_imp / dom / "improved_abstracts"
        new_dir = base_ai_imp / dom / "new_abstracts"
        rewritten_dir = base_ai_imp / dom / "rewritten_abstracts"

        if not paper_json_dir.exists():
            continue

        for paper_path in sorted(paper_json_dir.glob("W*.json")):
            if not paper_path.is_file():
                continue

            total_papers += 1

            # Enforce limit in terms of number of paper ids processed.
            if limit is not None and processed_papers >= limit:
                break

            with paper_path.open() as f:
                paper_obj = json.load(f)

            paper_id = paper_obj.get("id") or paper_path.stem
            domain = paper_obj.get("domain") or dom
            abstract = paper_obj.get("abstract")

            if not abstract:
                # No abstract to work with; skip this paper entirely.
                continue

            # Original abstract output
            out_dir_orig = ROOT / "humanization" / collection / domain / "original"
            out_dir_orig.mkdir(parents=True, exist_ok=True)
            orig_out = out_dir_orig / f"{paper_id}.json"

            if orig_out.exists() and not overwrite:
                skipped_original += 1
            else:
                payload = {
                    "paper_id": paper_id,
                    "domain": domain,
                    "abstract": abstract,
                }
                with orig_out.open("w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                written_original += 1

            # Helper to write a variant (improved/new/rewritten)
            def write_variant(
                variant_dir: Path,
                variant_name: str,
                written_counter_name: str,
                skipped_counter_name: str,
            ) -> None:
                nonlocal written_improved, written_new, written_rewritten
                nonlocal skipped_improved, skipped_new, skipped_rewritten

                variant_path = variant_dir / f"{paper_id}.json"
                if not variant_path.exists():
                    return

                with variant_path.open() as f:
                    variant_obj = json.load(f)
                var_abs = variant_obj.get("abstract")
                if not var_abs:
                    return

                out_dir_var = ROOT / "humanization" / collection / domain / variant_name
                out_dir_var.mkdir(parents=True, exist_ok=True)
                var_out = out_dir_var / f"{paper_id}.json"

                if var_out.exists() and not overwrite:
                    if skipped_counter_name == "improved":
                        skipped_improved += 1
                    elif skipped_counter_name == "new":
                        skipped_new += 1
                    elif skipped_counter_name == "rewritten":
                        skipped_rewritten += 1
                    return

                payload_var = {
                    "paper_id": paper_id,
                    "domain": domain,
                    "abstract": var_abs,
                }
                with var_out.open("w") as f:
                    json.dump(payload_var, f, ensure_ascii=False, indent=2)

                if written_counter_name == "improved":
                    written_improved += 1
                elif written_counter_name == "new":
                    written_new += 1
                elif written_counter_name == "rewritten":
                    written_rewritten += 1

            # Improved / new / rewritten variants (if available)
            write_variant(improved_dir, "improved", "improved", "improved")
            write_variant(new_dir, "new", "new", "new")
            write_variant(rewritten_dir, "rewritten", "rewritten", "rewritten")

            processed_papers += 1

        # Respect limit across domains
        if limit is not None and processed_papers >= limit:
            break

    print(f"Total papers found: {total_papers}")
    print(f"Paper ids processed: {processed_papers}")
    print(f"Written original abstracts: {written_original}")
    print(f"Written improved abstracts: {written_improved}")
    print(f"Written new abstracts: {written_new}")
    print(f"Written rewritten abstracts: {written_rewritten}")
    print(f"Skipped existing original: {skipped_original}")
    print(f"Skipped existing improved: {skipped_improved}")
    print(f"Skipped existing new: {skipped_new}")
    print(f"Skipped existing rewritten: {skipped_rewritten}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare per-paper originals and improved/new/rewritten abstracts for humanization.\n"
            "Reads from papers/{collection}/{domain}/paper_jsons/ and ai_improvement/{collection}/{domain}/*_abstracts,\n"
            "and writes JSON with only paper_id, domain, abstract to\n"
            "humanization/{collection}/{domain}/{original|improved|new|rewritten}/{paper_id}.json, skipping files that already exist."
        )
    )
    parser.add_argument(
        "--collection",
        default="2025_back_2023",
        help="Collection folder under papers/ (default: 2025_back_2023)",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        choices=["chemistry", "computer_science", "political_science", "theology"],
        help="Optional list of domains to restrict processing to.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in humanization/ if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of new humanization records to write (for testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        collection=args.collection,
        domains=args.domains,
        overwrite=args.overwrite,
        limit=args.limit,
    )

