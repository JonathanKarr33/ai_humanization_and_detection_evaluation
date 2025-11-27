import argparse
import json
from pathlib import Path

from humanizer import humanize_text


def process_folder(input_dir: Path, output_dir: Path) -> None:
    """Read plan JSON files, generate answers, and write them to parallel JSON files."""
    for json_file in input_dir.rglob("*.json"):
        # Preserve the relative path when writing output.
        relative_path = json_file.relative_to(input_dir)
        out_file = output_dir / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, str]] = json.loads(  # pyright: ignore [reportAny]
            json_file.read_text(encoding="utf-8")
        )
        results: list[dict[str, str]] = []
        for entry in entries:
            question: str = entry["question"]
            answer = entry["answer"]
            h_answer = humanize_text(answer)
            results.append({"question": question, "answer": h_answer})

        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanize responses from JSON files")
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing discipline/*.json files with questions and responses.",
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
