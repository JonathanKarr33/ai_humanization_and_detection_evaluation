#!/usr/bin/env python3
"""
Process all paper abstracts through PANGRAM API for AI detection.
Saves results to JSON with all PANGRAM fields, domain, and paper ID.
"""

import argparse
import json
from typing import Any

from pangram import Pangram

from config import CONFIG


def get_pangram_result(text: str, client: Pangram) -> dict[str, Any]:
    """
    Get AI detection result from PANGRAM API using the SDK.
    Returns the full API response as a dictionary, or None if failed.
    """
    assert text and text.strip()
    result = client.predict(text)
    return result


def process_abstracts(
    abstract_text: str, paper_id: str, domain: str, client: Pangram
) -> dict[str, Any]:
    # Get PANGRAM result
    pangram_result = get_pangram_result(abstract_text, client)
    assert pangram_result
    result_entry = {
        "paper_id": paper_id,
        "domain": domain,
        **pangram_result,
    }
    return result_entry


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process paper abstracts through PANGRAM API for AI detection."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    args = parser.parse_args()

    # Load input JSON
    with open(args.input, "r") as f:
        data = json.load(f)

    # Initialize PANGRAM client
    client = Pangram(api_key=CONFIG.AI_DETECTOR.API_KEY)

    # Process abstracts
    abstract_text = data["abstract"]
    paper_id = data["id"]
    domain = data["domain"]
    result = process_abstracts(abstract_text, paper_id, domain, client)

    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
