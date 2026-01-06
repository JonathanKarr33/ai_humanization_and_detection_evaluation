import csv
import json
from argparse import ArgumentParser
from collections.abc import MutableMapping
from typing import Any, cast

BANNED_FIELDS = frozenset({"text", "request_id", "prediction"})


def find_process(p: str):
    p_comps = p.replace("\\", "/").split("/")
    for c in p_comps:
        c = c.strip()
        if c.endswith("_pangram_results"):
            return c.removesuffix("_pangram_results").lower()
    assert False


# https://stackoverflow.com/a/6027615
def flatten(dictionary: dict[str, Any], parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(
                flatten(
                    cast(dict[str, Any], value), new_key, separator=separator
                ).items()
            )
        else:
            items.append((new_key, value))
    return cast(dict[str, Any], dict(items))


def filter_dict(i: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in i.items() if k not in BANNED_FIELDS}


parser = ArgumentParser(description="Process paper abstracts through AI for rewriting.")
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

        data_dict = filter_dict(flatten(data))
        fields = sorted(list(data_dict.keys())) + ["process"]
        process: str = find_process(file)
        data_dict["process"] = process
        if not writer:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
        writer.writerow(data_dict)
