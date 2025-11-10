import argparse
import json
from pathlib import Path

from aic import CLIENT
from config import CONFIG
from pydantic import BaseModel

# Prompt that asks the model to produce a short bullet‑point answer plan.
PLAN_SYSTEM_PROMPT = """You are an assistant that creates a concise answer plan for a layperson.
Given a question, provide a list of bullet points that outline the key steps or facts needed to answer it.
Return the result as a JSON object with a single key "plan" mapping to a list of strings.
Try to keep the points short and concise. Do not generate too many points.
The plan will then be used to write a 200 word paragraph of coherent text answering the question.
The plan should include provisions for both structure and content of the final answer.
Example output:
{{"plan": ["First point", "Second point", "Third point"]}}"""


class PlanResponse(BaseModel):
    plan: list[str]


def generate_plan(question: str) -> list[str]:
    """Ask the LLM to create a bullet‑point plan for the given question."""
    completion = CLIENT.chat.completions.parse(
        model=CONFIG.AI_ENDPOINT.MODEL_NAME,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        response_format=PlanResponse,
    )
    message = completion.choices[0].message
    assert message.parsed
    return list(message.parsed.plan)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    """Read each discipline's filtered questions, generate answer plans, and write JSON files."""
    for txt_file in input_dir.rglob("*.txt"):
        discipline = txt_file.stem  # e.g. "ml" from "./input_folder/ml.txt"

        questions = [
            line.strip()
            for line in txt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert questions

        results: list[dict[str, str | list[str]]] = []
        for q in questions:
            plan = generate_plan(q)
            results.append({"question": q, "plan": plan})

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{discipline}.json"
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate answer plans for filtered questions per discipline."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing discipline/*.txt files with one filtered question per line.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where JSON files with answer plans will be written.",
    )
    args = parser.parse_args()

    INPUT_FOLDER: Path = args.input_folder  # pyright: ignore [reportAny]
    OUTPUT_FOLDER: Path = args.output_folder  # pyright: ignore [reportAny]

    if not INPUT_FOLDER.is_dir():
        parser.error(
            f"Input folder '{INPUT_FOLDER}' does not exist or is not a directory."
        )

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
