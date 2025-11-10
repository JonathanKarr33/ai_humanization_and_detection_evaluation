import argparse
from pathlib import Path

from aic import CLIENT
from config import CONFIG
from pydantic import BaseModel

# System prompt template (moved to top of script)
SYSTEM_PROMPT_TEMPLATE = """You are an assistant that creates clear, concise questions for a layperson based on the content of a scientific paper. Try not to use scientific terms from the paper. Make sure that a layperson can answer these questions, make them simple. Generate exactly {n} questions that focus on the main topic of the paper. Do not refer to the paper in the text of the questions, they should self-contained and answerable even without reading the paper. Return the result as a JSON object with a single key "questions" mapping to a list of strings. Example output:
{{"questions": ["How can convolutional neural networks be used to classify images?", "How does the formation of perovskites change the material properties of steel?", "What is the role of ionic conductivity in the operation of a Li‑Ion battery?"]}}"""


# Pydantic model for the expected response
class QuestionsResponse(BaseModel):
    questions: list[str]


def generate_questions(paper_text: str, num_questions: int = 10) -> set[str]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(n=num_questions)

    completion = CLIENT.chat.completions.parse(
        model=CONFIG.AI_ENDPOINT.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": paper_text},
        ],
        temperature=0.7,
        response_format=QuestionsResponse,
    )

    message = completion.choices[0].message
    assert message.parsed
    parsed = message.parsed
    return set(parsed.questions)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    for txt_file in input_dir.rglob("*.txt"):
        # Determine discipline from the first part of the relative path
        try:
            discipline = txt_file.relative_to(input_dir).parts[0]
        except IndexError:
            # Skip files not following the expected <discipline>/<paper>.txt layout
            continue

        paper_text = txt_file.read_text(encoding="utf-8")
        questions = generate_questions(paper_text)

        if not questions:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        discipline_file = output_dir / f"{discipline}.txt"

        with discipline_file.open("a", encoding="utf-8") as f:
            for q in questions:
                _ = f.write(q.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate layperson‑friendly questions for papers in a directory."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the folder containing discipline/paper_name.txt files.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where discipline‑specific question files will be written.",
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
