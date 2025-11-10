import argparse
import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from aic import CLIENT
from config import CONFIG

# System prompt for generating a 200‑word answer based on a plan.
ANSWER_SYSTEM_PROMPT = """You are an assistant that writes a concise, coherent 200‑word answer to a question.
Use the provided answer plan as a guide, expanding each point into a flowing paragraph while staying within the word limit.
Return the result as a JSON object with a single key "answer" containing the generated text.
Do not include any additional commentary or formatting."""


class AnswerResponse(BaseModel):
    answer: str


def generate_answer(question: str, plan: list[str]) -> str:
    """Ask the LLM to write a 200‑word answer based on the question and plan."""
    plan_text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(plan))
    user_message = f"Question: {question}\n\nAnswer plan:\n{plan_text}"
    completion = CLIENT.chat.completions.parse(
        model=CONFIG.AI_ENDPOINT.MODEL_NAME,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format=AnswerResponse,
    )
    message = completion.choices[0].message
    assert message.parsed
    return message.parsed.answer


def process_folder(input_dir: Path, output_dir: Path) -> None:
    """Read plan JSON files, generate answers, and write them to parallel JSON files."""
    for json_file in input_dir.rglob("*.json"):
        # Preserve the relative path when writing output.
        relative_path = json_file.relative_to(input_dir)
        out_file = output_dir / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, str | list[str]]] = json.loads(  # pyright: ignore [reportAny]
            json_file.read_text(encoding="utf-8")
        )
        results: list[dict[str, str]] = []
        for entry in entries:
            question: str = cast(str, entry["question"])
            plan: list[str] = cast(list[str], entry["plan"])
            answer = generate_answer(question, plan)
            results.append({"question": question, "answer": answer})

        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 200‑word answers from plan JSON files."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing discipline/*.json files with questions and plans.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where answer JSON files will be written, mirroring the input structure.",
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
