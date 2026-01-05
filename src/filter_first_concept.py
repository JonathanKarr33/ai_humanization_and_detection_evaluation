#!/usr/bin/env python3
"""
Filter papers where theology or political science is the #1 concept.
Copy filtered papers to domain/first/ subfolders.
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# OpenAlex concept IDs
CONCEPTS = {
    "political_science": "C17744445",
    "theology": "C27206212"
}

OPENALEX_API = "https://api.openalex.org/works"
EMAIL = os.getenv("OPENALEX_EMAIL", "jkarr@nd.edu")

api_headers = {
    "User-Agent": f"PaperFilter ({EMAIL})",
    "Accept": "application/json"
}

PAPERS_DIR = Path("papers")
METADATA_FILE = PAPERS_DIR / "metadata.jsonl"


def get_paper_concepts(openalex_id: str) -> Optional[list]:
    """Fetch paper from OpenAlex API and return its concepts list."""
    # Extract work ID from openalex_id URL if it's a full URL
    work_id = openalex_id
    if openalex_id.startswith("https://openalex.org/"):
        work_id = openalex_id.replace("https://openalex.org/", "")
    
    url = f"{OPENALEX_API}/{work_id}"
    
    try:
        response = requests.get(url, headers=api_headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("concepts", [])
    except Exception as e:
        print(f"Error fetching {openalex_id}: {e}")
        return None


def is_domain_first_concept(concepts: list, domain: str) -> bool:
    """Check if the domain concept is ranked #1 (highest score)."""
    if not concepts:
        return False
    
    concept_id = CONCEPTS.get(domain)
    if not concept_id:
        return False
    
    # Concepts are sorted by relevance score (highest first)
    # Check if the first concept matches our domain
    if concepts and len(concepts) > 0:
        first_concept = concepts[0]
        first_concept_id = first_concept.get("id", "")
        if concept_id in first_concept_id or first_concept_id.endswith(concept_id):
            return True
    
    return False


def copy_paper_to_first_folder(paper_id: str, domain: str) -> None:
    """Copy paper files (PDF, text, abstract) to domain/first/ subfolder."""
    first_dir = PAPERS_DIR / domain / "first"
    first_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy PDF if it exists
    pdf_source = PAPERS_DIR / domain / "pdfs" / f"{paper_id}.pdf"
    if pdf_source.exists():
        pdf_dest = first_dir / "pdfs"
        pdf_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_source, pdf_dest / f"{paper_id}.pdf")
    
    # Copy text if it exists
    text_source = PAPERS_DIR / domain / "text" / f"{paper_id}.txt"
    if text_source.exists():
        text_dest = first_dir / "text"
        text_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(text_source, text_dest / f"{paper_id}.txt")
    
    # Copy abstract if it exists
    abstract_source = PAPERS_DIR / domain / "abstracts" / f"{paper_id}.txt"
    if abstract_source.exists():
        abstract_dest = first_dir / "abstracts"
        abstract_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abstract_source, abstract_dest / f"{paper_id}.txt")


def main():
    """Main filtering pipeline."""
    if not METADATA_FILE.exists():
        print(f"Error: {METADATA_FILE} not found")
        return
    
    # Read all papers from metadata
    papers = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    
    print(f"Found {len(papers)} papers in metadata")
    
    # Filter papers where domain is #1 concept
    filtered_papers = []
    
    for paper in tqdm(papers, desc="Checking concept rankings"):
        domain = paper.get("domain")
        openalex_id = paper.get("openalex_id")
        paper_id = paper.get("id")
        
        if not domain or not openalex_id or not paper_id:
            continue
        
        if domain not in CONCEPTS:
            continue
        
        # Fetch concepts from OpenAlex
        concepts = get_paper_concepts(openalex_id)
        if concepts is None:
            continue
        
        # Check if domain is #1 concept
        if is_domain_first_concept(concepts, domain):
            filtered_papers.append(paper)
            # Copy paper to first/ subfolder
            copy_paper_to_first_folder(paper_id, domain)
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"\nFiltered {len(filtered_papers)} papers where domain is #1 concept")
    
    # Create "first" marker files in each domain/first/ folder
    for domain in CONCEPTS.keys():
        first_dir = PAPERS_DIR / domain / "first"
        if first_dir.exists():
            marker_file = first_dir / "first.txt"
            with open(marker_file, "w", encoding="utf-8") as f:
                f.write("first\n")
            print(f"Created marker file: {marker_file}")
    
    # Save filtered metadata
    filtered_metadata_file = PAPERS_DIR / "metadata_first.jsonl"
    with open(filtered_metadata_file, "w", encoding="utf-8") as f:
        for paper in filtered_papers:
            f.write(json.dumps(paper) + "\n")
    
    print(f"Saved filtered metadata to: {filtered_metadata_file}")


if __name__ == "__main__":
    main()
