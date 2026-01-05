#!/usr/bin/env python3
"""
Test text files using PANGRAM API for AI detection.
PANGRAM is an AI detector that returns likelihood scores for text.
"""
import json
import csv
import os
import re
import requests
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

THEO_PAPERS_DIR = Path("theo_papers")
PANGRAM_CACHE_FILE = Path("pangram_cache.json")
PANGRAM_RESULTS_FILE = Path("pangram_results.csv")
PANGRAM_API_KEY = os.getenv("PANGRAM_API")
PANGRAM_API_URL = os.getenv("PANGRAM_API_URL", "https://api.pangram.com/v1/detect")


def get_pangram_likelihood(text: str) -> Optional[float]:
    """
    Get AI detection likelihood from PANGRAM API.
    Returns a float between 0 and 1, where 1.0 means likely AI-generated.
    """
    if not PANGRAM_API_KEY:
        raise ValueError("PANGRAM_API key not found in environment variables")
    
    if not text or not text.strip():
        return 0.0
    
    try:
        # Make API request to PANGRAM
        headers = {
            "Authorization": f"Bearer {PANGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }
        
        response = requests.post(PANGRAM_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        # Extract likelihood from response (adjust field name based on actual API response)
        # Common field names: "likelihood", "score", "ai_probability", "probability"
        likelihood = data.get("likelihood") or data.get("score") or data.get("ai_probability") or data.get("probability", 0.0)
        
        return float(likelihood)
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error getting likelihood: {e}")
        return None


def process_theo_papers() -> Dict[str, float]:
    """Process all theo_papers files and get PANGRAM AI detection likelihoods."""
    cache = {}
    
    if not THEO_PAPERS_DIR.exists():
        print(f"Error: {THEO_PAPERS_DIR} directory not found")
        return cache
    
    # Load existing cache to avoid re-processing
    if PANGRAM_CACHE_FILE.exists():
        try:
            with open(PANGRAM_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} entries from cache")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    
    # Process each paper directory (one, two, three, four, five)
    for paper_dir in sorted(THEO_PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        
        paper_name = paper_dir.name
        
        # Process raw.txt, rewritten_raw.txt, and fully_rewritten.txt
        for filename in ["raw.txt", "rewritten_raw.txt", "fully_rewritten.txt"]:
            file_path = paper_dir / filename
            file_path_str = str(file_path.absolute())
            
            # Skip if already in cache
            if file_path_str in cache:
                print(f"Skipping {file_path} (already in cache)")
                continue
            
            if file_path.exists():
                try:
                    text = file_path.read_text(encoding="utf-8")
                    print(f"Processing {file_path}...")
                    likelihood = get_pangram_likelihood(text)
                    
                    if likelihood is not None:
                        cache[file_path_str] = likelihood
                        print(f"  Likelihood: {likelihood:.10f}")
                    else:
                        print(f"  Failed to get likelihood")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return cache


def save_cache(cache: Dict[str, float]) -> None:
    """Save cache to JSON file."""
    with open(PANGRAM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\nSaved cache to {PANGRAM_CACHE_FILE}")


def generate_results_csv(cache: Dict[str, float]) -> None:
    """Generate CSV results file from cache."""
    # Group by paper number (one, two, three, four, five)
    results = {}
    
    for file_path_str, likelihood in cache.items():
        # Extract paper number and file type from path
        # e.g., "theo_papers/one/raw.txt" -> paper="one", type="raw"
        match = re.search(r'theo_papers/(\w+)/(\w+)\.txt', file_path_str)
        if match:
            paper_num = match.group(1)
            file_type = match.group(2)
            
            if paper_num not in results:
                results[paper_num] = {}
            
            # Map file types to CSV columns
            if file_type == "raw":
                results[paper_num]["raw_likelihood"] = likelihood
            elif file_type == "rewritten_raw":
                results[paper_num]["rewritten_raw_likelihood"] = likelihood
            elif file_type == "fully_rewritten":
                results[paper_num]["fully_likelihood"] = likelihood
    
    # Write CSV
    with open(PANGRAM_RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_number", "raw_likelihood", 
                                                "rewritten_raw_likelihood", "fully_likelihood"])
        writer.writeheader()
        
        for paper_num in sorted(results.keys()):
            row = {
                "paper_number": paper_num,
                "raw_likelihood": results[paper_num].get("raw_likelihood", 0.0),
                "rewritten_raw_likelihood": results[paper_num].get("rewritten_raw_likelihood", 0.0),
                "fully_likelihood": results[paper_num].get("fully_likelihood", 0.0)
            }
            writer.writerow(row)
    
    print(f"Saved results to {PANGRAM_RESULTS_FILE}")


def main():
    """Main function."""
    if not PANGRAM_API_KEY:
        print("Error: PANGRAM_API key not found in .env file")
        print("Please add PANGRAM_API=your_key to .env file")
        return
    
    print("Testing text files with PANGRAM AI detector...\n")
    
    # Process files
    cache = process_theo_papers()
    
    # Save cache and generate CSV
    if cache:
        save_cache(cache)
        generate_results_csv(cache)
        print("\n✅ Done!")
    else:
        print("\n⚠️  No results to save")


if __name__ == "__main__":
    main()
