#!/usr/bin/env python3
"""
Parse abstracts and full text from scraped papers, organized by domain.
Does not refetch PDFs - only processes existing data.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional


# ==========================
# CONFIG
# ==========================
BASE_DIR = Path("papers")
METADATA_FILE = BASE_DIR / "metadata.jsonl"
OUTPUT_BASE = Path("parsed_papers")
# Source structure: papers/{domain}/pdfs/, papers/{domain}/text/, papers/{domain}/abstracts/

# Domain folders
DOMAIN_DIRS = {
    "political_science": OUTPUT_BASE / "political_science",
    "theology": OUTPUT_BASE / "theology"
}


def ensure_dirs():
    """Create output directories for each domain."""
    for domain_dir in DOMAIN_DIRS.values():
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / "abstracts").mkdir(exist_ok=True)
        (domain_dir / "fulltext").mkdir(exist_ok=True)


def parse_metadata() -> Dict[str, dict]:
    """Parse metadata.jsonl and return a dict mapping paper_id to metadata."""
    papers = {}
    
    if not METADATA_FILE.exists():
        print(f"Warning: {METADATA_FILE} not found")
        return papers
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                metadata = json.loads(line)
                paper_id = metadata.get("id")
                if paper_id:
                    papers[paper_id] = metadata
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in metadata: {e}")
                continue
    
    return papers


def get_abstract(paper_id: str, domain: str) -> Optional[str]:
    """Read abstract from domain-specific abstracts folder if it exists, fallback to metadata."""
    # Domain-specific folder: papers/{domain}/abstracts/
    abstract_path = BASE_DIR / domain / "abstracts" / f"{paper_id}.txt"
    
    if abstract_path.exists():
        try:
            with open(abstract_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                return text if text else None
        except Exception as e:
            print(f"Warning: Failed to read abstract file for {paper_id}: {e}")
    
    return None


def get_fulltext(paper_id: str, domain: str) -> Optional[str]:
    """Read full text from domain-specific text folder if it exists."""
    # Domain-specific folder: papers/{domain}/text/
    text_path = BASE_DIR / domain / "text" / f"{paper_id}.txt"
    
    if not text_path.exists():
        return None
    
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text if text else None
    except Exception as e:
        print(f"Warning: Failed to read text for {paper_id}: {e}")
        return None


def save_abstract(domain: str, paper_id: str, abstract: str):
    """Save abstract to domain-specific folder."""
    if not abstract or not abstract.strip():
        return
    
    domain_dir = DOMAIN_DIRS.get(domain)
    if not domain_dir:
        print(f"Warning: Unknown domain {domain}, skipping")
        return
    
    abstract_path = domain_dir / "abstracts" / f"{paper_id}.txt"
    try:
        with open(abstract_path, "w", encoding="utf-8") as f:
            f.write(abstract.strip())
    except Exception as e:
        print(f"Warning: Failed to save abstract for {paper_id}: {e}")


def save_fulltext(domain: str, paper_id: str, fulltext: str):
    """Save full text to domain-specific folder."""
    if not fulltext or not fulltext.strip():
        return
    
    domain_dir = DOMAIN_DIRS.get(domain)
    if not domain_dir:
        print(f"Warning: Unknown domain {domain}, skipping")
        return
    
    fulltext_path = domain_dir / "fulltext" / f"{paper_id}.txt"
    try:
        with open(fulltext_path, "w", encoding="utf-8") as f:
            f.write(fulltext.strip())
    except Exception as e:
        print(f"Warning: Failed to save fulltext for {paper_id}: {e}")


def main():
    """Main parsing pipeline."""
    print("📚 Parsing papers by domain...\n")
    
    ensure_dirs()
    
    # Parse metadata
    print("Reading metadata...")
    papers = parse_metadata()
    print(f"Found {len(papers)} papers in metadata\n")
    
    # Process each paper
    stats = {
        "political_science": {"abstracts": 0, "fulltext": 0},
        "theology": {"abstracts": 0, "fulltext": 0}
    }
    
    for paper_id, metadata in papers.items():
        domain = metadata.get("domain")
        if not domain or domain not in DOMAIN_DIRS:
            continue
        
        # Process abstract - prefer from domain-specific abstracts folder, fallback to metadata
        abstract = get_abstract(paper_id, domain)
        if not abstract:
            abstract = metadata.get("abstract")
        
        if abstract:
            save_abstract(domain, paper_id, abstract)
            stats[domain]["abstracts"] += 1
        
        # Process full text
        fulltext = get_fulltext(paper_id, domain)
        if fulltext:
            save_fulltext(domain, paper_id, fulltext)
            stats[domain]["fulltext"] += 1
    
    # Print summary
    print("\n✅ Parsing complete!\n")
    print("Summary:")
    for domain, counts in stats.items():
        print(f"  {domain}:")
        print(f"    Abstracts: {counts['abstracts']}")
        print(f"    Full text: {counts['fulltext']}")
    
    print(f"\n📁 Output saved to: {OUTPUT_BASE.absolute()}")


if __name__ == "__main__":
    main()
