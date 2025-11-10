import argparse
import os
import sys

import pymupdf


def convert_pdf_to_txt(pdf_path: str, txt_path: str):
    print("Converting", pdf_path, "->", txt_path, file=sys.stdout, flush=True)
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += str(page.get_text())  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)


def process_folder(input_root: str, output_root: str):
    for dirpath, _, filenames in os.walk(input_root):
        rel_path = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, rel_path)
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                in_file = os.path.join(dirpath, filename)
                out_file = os.path.join(out_dir, os.path.splitext(filename)[0] + ".txt")
                convert_pdf_to_txt(in_file, out_file)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to TXT preserving folder structure."
    )
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_folder", help="Path to the output folder")
    args = parser.parse_args()

    INPUT_FOLDER = str(args.input_folder)  # pyright: ignore [reportAny]
    OUTPUT_FOLDER = str(args.output_folder)  # pyright: ignore [reportAny]

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
