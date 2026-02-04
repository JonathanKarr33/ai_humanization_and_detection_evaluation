#!/usr/bin/env python3
"""
Process all paper abstracts through PANGRAM API for AI detection.
Saves results to JSON with all PANGRAM fields, domain, and paper ID.
"""

import argparse
import json
from typing import Annotated, Any

import requests
from pydantic import BaseModel, Field

from aic import LLM_AI_DETECTION_CLIENT
from config import CONFIG


class LLMAiDetectionResult(BaseModel):
    result_explanation: str = Field(
        description="Explanation of why this probability was chosen. 1-2 short sentences"
    )

    ai_probability: Annotated[
        int,
        Field(
            strict=True,
            gt=0,
            le=100,
            description="Probability that the input text is AI-generated, percentage (0-100)",
        ),
    ]


LLM_AI_DETECTION_PROMPT = (
    """
You are an expert scientist who excels in detecting AI in texts. Your task is to check whether the text given to you in the user's message is AI-generated or not. Please provide a probability that the text is AI generated (whole number, between 0 and 100) and an explanation of why you chose this number. Please respond in JSON format, following this schema:
""".strip()
    + "\n"
    + json.dumps(LLMAiDetectionResult.model_json_schema())
)


def get_result(text: str) -> dict[str, Any]:
    assert text and text.strip()
    text = text.strip()

    response = LLM_AI_DETECTION_CLIENT.chat.completions.parse(
        messages=[
            {"role": "system", "content": LLM_AI_DETECTION_PROMPT},
            {"role": "user", "content": text},
        ],
        model=CONFIG.LLM_BASED_DETECTOR.MODEL_NAME,
        response_format=LLMAiDetectionResult,
    )
    assert response.choices and len(response.choices) > 0
    content = response.choices[0].message.parsed
    assert content is not None

    d = {
        "ai_probability": content.ai_probability / 100,
        "explanation": content.result_explanation,
        "model_name": CONFIG.LLM_BASED_DETECTOR.MODEL_NAME,
    }
    return d


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
