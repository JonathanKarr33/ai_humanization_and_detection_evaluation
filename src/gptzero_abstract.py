#!/usr/bin/env python3
"""
Process all paper abstracts through PANGRAM API for AI detection.
Saves results to JSON with all PANGRAM fields, domain, and paper ID.
"""

import argparse
import json
from typing import Any

import requests

from config import CONFIG

GPTZERO_ENDPOINT = "https://api.gptzero.me/v2/predict/text"


def get_result(text: str) -> dict[str, Any]:
    assert text and text.strip()
    text = text.strip()

    payload = {"document": text, "multilingual": False}

    headers = {
        "x-api-key": CONFIG.GPT_ZERO_DETECTOR.API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    response = requests.post(GPTZERO_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    response = response.json()

    docs_res = response["documents"]
    assert len(docs_res) == 1
    res = docs_res[0]

    result = res["class_probabilities"]
    result["document_class"] = res["document_classification"]
    return result


def process_abstracts(
    abstract_text: str,
    paper_id: str,
    domain: str,
) -> dict[str, Any]:
    detection_result = get_result(abstract_text)
    assert detection_result
    result_entry = {
        "paper_id": paper_id,
        "domain": domain,
        **detection_result,
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

    # Process abstracts
    abstract_text = data["abstract"]
    paper_id = data["id"]
    domain = data["domain"]
    result = process_abstracts(abstract_text, paper_id, domain)

    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
