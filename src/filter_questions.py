import argparse
from pathlib import Path

from aic import CLIENT
from config import CONFIG
from pydantic import BaseModel

# System prompt to ask the model to keep only the two best, most self‑contained,
# and easiest‑to‑answer questions.
FILTER_SYSTEM_PROMPT = """You are an assistant that selects the two best, most self‑contained, and easiest to answer questions from a given list. The selected questions should be understandable without any external context (i.e. should not mention "this test" or "this paper") and should be simple for a layperson to answer. Return only the selected questions as a JSON object with a single key "questions" mapping to a list of exactly two strings. Example output:
{{"questions": ["Why does water boil at 100 °C?", "What causes a rainbow to form?"]}}"""


class FilterResponse(BaseModel):
    questions: list[str]


def select_top_two(questions: list[str]) -> list[str]:
    """Send the list of questions to the LLM and retrieve the two best ones."""
    user_content = "\n".join(f"- {q}" for q in questions)

    completion = CLIENT.chat.completions.parse(
        model=CONFIG.AI_ENDPOINT.MODEL_NAME,
        messages=[
            {"role": "system", "content": FILTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        response_format=FilterResponse,
    )
    message = completion.choices[0].message
    assert message.parsed
    return list(message.parsed.questions)


def process_questions_folder(input_dir: Path, output_dir: Path) -> None:
    for txt_file in input_dir.rglob("*.txt"):
        # Discipline is the basename of the .txt file without extension (e.g., ./folder/ml.txt → ml)
        discipline = txt_file.stem

        raw_questions = [
            line.strip()
            for line in txt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not raw_questions:
            continue

        best_two = select_top_two(raw_questions)
        if not best_two:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{discipline}.txt"
        with out_file.open("a", encoding="utf-8") as f:
            for q in best_two:
                print(q, file=f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter generated questions to the two best per discipline."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing discipline/*.txt files with one question per line.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where filtered questions will be written.",
    )
    args = parser.parse_args()

    INPUT_FOLDER: Path = args.input_folder  # pyright: ignore [reportAny]
    OUTPUT_FOLDER: Path = args.output_folder  # pyright: ignore [reportAny]

    if not INPUT_FOLDER.is_dir():
        parser.error(
            f"Input folder '{INPUT_FOLDER}' does not exist or is not a directory."
        )

    process_questions_folder(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
