#!/usr/bin/env python3
"""
Process all paper abstracts through PANGRAM API for AI detection.
Saves results to JSON with all PANGRAM fields, domain, and paper ID.
"""
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from pangram import Pangram

# Load environment variables from .env file
load_dotenv()

PAPERS_DIR = Path("papers")
OUTPUT_JSON = Path("pangram_abstracts_results.json")
PANGRAM_API_KEY = os.getenv("PANGRAM_API")


def get_pangram_result(text: str, client: Pangram) -> Optional[Dict]:
    """
    Get AI detection result from PANGRAM API using the SDK.
    Returns the full API response as a dictionary, or None if failed.
    """
    if not text or not text.strip():
        return None
    
    try:
        # Use the SDK's predict method (uses extended model by default)
        result = client.predict(text)
        # Return the full response
        return result
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        return None


def load_existing_results() -> Dict[str, Dict]:
    """Load existing results from JSON file."""
    if not OUTPUT_JSON.exists():
        return {}
    
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert list to dict if needed, or return dict as-is
            if isinstance(data, list):
                return {item.get("paper_id"): item for item in data if item.get("paper_id")}
            elif isinstance(data, dict):
                return data
            else:
                return {}
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return {}


def save_results(results: Dict[str, Dict]) -> None:
    """Save results to JSON file."""
    # Convert dict to list for easier JSON handling
    results_list = list(results.values())
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(results_list)} results to {OUTPUT_JSON}")


def get_all_abstracts(test_mode: bool = False, limit: int = 10) -> List[Dict]:
    """
    Get all abstracts from papers directory.
    Returns list of dicts with: paper_id, domain, abstract_text
    """
    abstracts = []
    
    if not PAPERS_DIR.exists():
        print(f"Error: {PAPERS_DIR} directory not found")
        return abstracts
    
    # Get all domain directories
    for domain_dir in PAPERS_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain = domain_dir.name
        abstracts_dir = domain_dir / "abstracts"
        
        if not abstracts_dir.exists():
            continue
        
        # Get all abstract files
        for abstract_file in abstracts_dir.glob("*.txt"):
            paper_id = abstract_file.stem  # e.g., "W1580878179" from "W1580878179.txt"
            
            try:
                abstract_text = abstract_file.read_text(encoding="utf-8").strip()
                if abstract_text:
                    abstracts.append({
                        "paper_id": paper_id,
                        "domain": domain,
                        "abstract_text": abstract_text
                    })
            except Exception as e:
                print(f"Warning: Could not read {abstract_file}: {e}")
    
    # Sort by paper_id for consistent ordering
    abstracts.sort(key=lambda x: x["paper_id"])
    
    if test_mode:
        abstracts = abstracts[:limit]
        print(f"Test mode: Processing first {len(abstracts)} abstracts")
    
    return abstracts


def process_abstracts(test_mode: bool = False, limit: int = 10) -> None:
    """Process all abstracts through PANGRAM API."""
    print("Processing abstracts with PANGRAM AI detector...\n")
    
    # Load existing results
    existing_results = load_existing_results()
    print(f"Loaded {len(existing_results)} existing results from {OUTPUT_JSON}")
    
    # Get all abstracts
    abstracts = get_all_abstracts(test_mode=test_mode, limit=limit)
    print(f"Found {len(abstracts)} abstracts to process\n")
    
    if not abstracts:
        print("No abstracts found to process.")
        return
    
    # Filter out already processed abstracts
    to_process = [a for a in abstracts if a["paper_id"] not in existing_results]
    skipped = len(abstracts) - len(to_process)
    
    if skipped > 0:
        print(f"Skipping {skipped} abstracts already in results")
    
    if not to_process:
        print("All abstracts already processed!")
        return
    
    print(f"Processing {len(to_process)} abstracts...\n")
    
    # Initialize PANGRAM client
    if not PANGRAM_API_KEY:
        print("Error: PANGRAM_API key not found in .env file")
        return
    
    try:
        client = Pangram(api_key=PANGRAM_API_KEY)
        print("✅ PANGRAM client initialized\n")
    except Exception as e:
        print(f"Error initializing PANGRAM client: {e}")
        return
    
    # Process each abstract
    results = existing_results.copy()
    first_error_shown = False
    save_interval = 10  # Save every 10 results
    processed_count = 0
    
    for abstract_data in tqdm(to_process, desc="Processing", unit="abstract"):
        paper_id = abstract_data["paper_id"]
        domain = abstract_data["domain"]
        abstract_text = abstract_data["abstract_text"]
        
        # Get PANGRAM result
        pangram_result = get_pangram_result(abstract_text, client)
        
        if pangram_result:
            # Create result entry with all PANGRAM fields plus our metadata
            result_entry = {
                "paper_id": paper_id,
                "domain": domain,
                **pangram_result  # Include all fields from PANGRAM response
            }
            results[paper_id] = result_entry
        else:
            # Still save entry with error indicator
            result_entry = {
                "paper_id": paper_id,
                "domain": domain,
                "error": "Failed to get PANGRAM result"
            }
            results[paper_id] = result_entry
            
            # Show detailed error only for first failure
            if not first_error_shown:
                print(f"\n⚠️  First failure detected for {paper_id}")
                print(f"   Check the error message above for details.")
                first_error_shown = True
        
        processed_count += 1
        
        # Save incrementally every N results to avoid losing progress
        if processed_count % save_interval == 0:
            save_results(results)
            # Use tqdm.write to avoid interfering with progress bar
            tqdm.write(f"💾 Saved progress: {len(results)} results")
    
    # Final save
    save_results(results)
    
    # Print summary
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = len(results) - successful
    
    print(f"\n✅ Done!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(results)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process paper abstracts through PANGRAM API for AI detection."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first 10 abstracts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of abstracts to process in test mode (default: 10)"
    )
    args = parser.parse_args()
    
    process_abstracts(test_mode=args.test, limit=args.limit)


if __name__ == "__main__":
    main()
