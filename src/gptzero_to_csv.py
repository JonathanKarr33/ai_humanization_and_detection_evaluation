import csv
import json
from argparse import ArgumentParser
from typing import Any, cast


def find_process(p: str):
    p_comps = p.replace("\\", "/").split("/")
    for c in p_comps:
        c = c.strip()
        if c.endswith("_gptzero_results"):
            return c.removesuffix("_gptzero_results").lower()
    assert False


FIELD_MAPPING = {
    "ai": "probability_ai",
    "human": "probability_human",
    "mixed": "probability_mixed",
}


def rename_fields(d: dict[str, Any]):
    ks = list(d.keys())
    for k in ks:
        if k in FIELD_MAPPING:
            d[FIELD_MAPPING[k]] = d.pop(k)
    return d


parser = ArgumentParser()
parser.add_argument(
    "--input", type=str, required=True, help="Paths to pangram JSON files", nargs="+"
)
parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
args = parser.parse_args()


input_files = cast(list[str], args.input)
writer: csv.DictWriter[Any] | None = None
with open(cast(str, args.output), "w", newline="") as csvfile:
    for file in input_files:
        if not file.endswith(".json"):
            continue
        with open(file, "r") as f:
            data = json.load(f)

        data_dict = rename_fields(data)
        fields = [
            "paper_id",
            "process",
            "domain",
            "probability_ai",
            "probability_mixed",
            "probability_human",
            "document_class",
        ]
        process: str = find_process(file)
        data_dict["process"] = process
        if not writer:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
        writer.writerow(data_dict)
