from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from variants import VARIANTS


ROOT = Path(__file__).resolve().parents[1]

REWRITE_VARIANTS = [v for v in VARIANTS if v != "original"]


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
    Stage humanization inputs (no Undetectable API call).

    - Reads originals from papers/{collection}/{domain}/paper_jsons/W*.json
    - Reads rewrites from ai_improvement/{collection}/{domain}/{variant}/W*.json
    - Writes JSONs to humanization/{collection}/{domain}/{variant}/{paper_id}.json
    """
    base_papers = ROOT / "papers" / collection
    base_ai_imp = ROOT / "ai_improvement" / collection

    total_papers = 0
    processed_papers = 0
    written: dict[str, int] = {v: 0 for v in VARIANTS}
    skipped: dict[str, int] = {v: 0 for v in VARIANTS}

    for dom in _domains_iter(domains):
        paper_json_dir = base_papers / dom / "paper_jsons"
        if not paper_json_dir.exists():
            continue

        ai_dirs = {variant: base_ai_imp / dom / variant for variant in REWRITE_VARIANTS}

        for paper_path in sorted(paper_json_dir.glob("W*.json")):
            if not paper_path.is_file():
                continue

            total_papers += 1
            if limit is not None and processed_papers >= limit:
                break

            with paper_path.open() as f:
                paper_obj = json.load(f)

            paper_id = paper_obj.get("id") or paper_path.stem
            domain = paper_obj.get("domain") or dom
            abstract = paper_obj.get("abstract")
            if not abstract:
                continue

            out_dir_orig = ROOT / "humanization" / collection / domain / "original"
            out_dir_orig.mkdir(parents=True, exist_ok=True)
            orig_out = out_dir_orig / f"{paper_id}.json"
            if orig_out.exists() and not overwrite:
                skipped["original"] += 1
            else:
                with orig_out.open("w") as f:
                    json.dump(
                        {"paper_id": paper_id, "domain": domain, "abstract": abstract},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                written["original"] += 1

            def write_variant(variant: str, variant_path: Path) -> None:
                if not variant_path.exists():
                    return
                with variant_path.open() as f:
                    variant_obj = json.load(f)
                var_abs = variant_obj.get("abstract")
                if not var_abs:
                    return

                out_dir_var = ROOT / "humanization" / collection / domain / variant
                out_dir_var.mkdir(parents=True, exist_ok=True)
                var_out = out_dir_var / f"{paper_id}.json"
                if var_out.exists() and not overwrite:
                    skipped[variant] += 1
                    return
                with var_out.open("w") as f:
                    json.dump(
                        {"paper_id": paper_id, "domain": domain, "abstract": var_abs},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                written[variant] += 1

            for variant in REWRITE_VARIANTS:
                write_variant(variant, ai_dirs[variant] / f"{paper_id}.json")

            processed_papers += 1

        if limit is not None and processed_papers >= limit:
            break

    print(f"Total papers found: {total_papers}")
    print(f"Paper ids processed: {processed_papers}")
    for variant in VARIANTS:
        print(f"Written {variant}: {written[variant]}")
        print(f"Skipped existing {variant}: {skipped[variant]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare per-paper originals and rewrite abstracts for humanization.\n"
            "Reads from papers/ and ai_improvement/, writes humanization/{collection}/{domain}/{variant}/."
        )
    )
    parser.add_argument("--collection", default="2025_back_2023")
    parser.add_argument(
        "--domains",
        nargs="*",
        choices=["chemistry", "computer_science", "political_science", "theology"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        collection=args.collection,
        domains=args.domains,
        overwrite=args.overwrite,
        limit=args.limit,
    )
