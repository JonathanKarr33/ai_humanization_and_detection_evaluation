import argparse
import json
from pathlib import Path
from typing import cast

from pangram import Pangram

from config import CONFIG


def process_folder(input_dir: Path, output_dir: Path) -> None:
    """Read plan JSON files, generate answers, and write them to parallel JSON files."""

    pangram_client = Pangram(api_key=CONFIG.AI_DETECTOR.API_KEY)

    for json_file in input_dir.rglob("*.json"):
        # Preserve the relative path when writing output.
        relative_path = json_file.relative_to(input_dir)
        out_file = output_dir / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, str]] = json.loads(  # pyright: ignore [reportAny]
            json_file.read_text(encoding="utf-8")
        )
        results: list[dict[str, str | dict[str, float | str]]] = []
        for entry in entries:
            question: str = entry["question"]
            answer = entry["answer"]
            pangram_result = cast(
                dict[str, float | str],
                pangram_client.predict_extended(answer),  # pyright: ignore [reportUnknownMemberType]
            )
            ai_lik = float(pangram_result["avg_ai_likelihood"])
            str_result = str(pangram_result["prediction"])
            str_result_short = str(pangram_result["prediction_short"])
            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "ai_detection": {
                        "likelihood": ai_lik,
                        "text": str_result,
                        "short_text": str_result_short,
                    },
                }
            )

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
