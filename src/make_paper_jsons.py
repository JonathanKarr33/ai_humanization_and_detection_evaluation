import json
import sys
from argparse import ArgumentParser
from typing import cast

assert __name__ == "__main__", "Only run as script"

aprs = ArgumentParser()
aprs.add_argument("--papers", type=str, required=True, help="Input folder path")
aprs.add_argument("--metadata", type=str, required=True, help="Paper metadata file")
aprs.add_argument("--output", type=str, required=True, help="Output file path")
args = aprs.parse_args()


with open(cast(str, args.metadata), "r", encoding="utf-8") as metafile:
    for line in metafile:
        data = cast(dict[str, str], json.loads(line))
        paper_domain = data["domain"]
        paper_id = data["id"]

        paper_text_path = f"{cast(str, args.papers)}/{paper_domain}/text/{paper_id}.txt"
        paper_abs_path = (
            f"{cast(str, args.papers)}/{paper_domain}/abstracts/{paper_id}.txt"
        )

        try:
            with open(paper_text_path, "r", encoding="utf-8") as textfile:
                text = textfile.read().strip()
        except FileNotFoundError:
            print("Full text for paper with id", paper_id, "not found", file=sys.stderr)
            continue

        try:
            with open(paper_abs_path, "r", encoding="utf-8") as absfile:
                abstract = absfile.read().strip()
        except FileNotFoundError:
            print("Abstract for paper with id", paper_id, "not found", file=sys.stderr)
            continue

        new_data = {
            "abstract": abstract,
            "text": text,
        }

        full_data = data | new_data

        with open(
            f"{cast(str, args.output)}/{paper_id}.txt", "x", encoding="utf-8"
        ) as outfile:
            json.dump(full_data, outfile, ensure_ascii=True, indent=2)
            outfile.write("\n")

        print(".", end="", flush=True, file=sys.stderr)
print("", file=sys.stderr)
