import json
from argparse import ArgumentParser
from collections.abc import Callable
from typing import cast

from aic import CLIENT
from config import CONFIG

REWRITING_SYSTEM_PROMPT_MODE_ABSTRACT_ONLY = """
You are an expert scientist who excels in writing papers. Your task is to rewrite a given paper abstract to sound better and more human-like. The abstract will be given in the next message. Please only return the rewritten text, do not return any intructions or anything like that. Only return the rewritten text. Make sure to make at least some edits or changes to the text.
""".strip()

REWRITING_SYSTEM_PROMPT_MODE_NEW_ABSTRACT = """
You are an expert scientist who excels in writing papers. Your task is to write a naturally sounding abstract for a paper that will be given to you. The full will be given in the next message. Please only return the written abstract, do not return any intructions or anything like that. Only return the written text. Make sure to not copy the text of the paper verbatim.
""".strip()

REWRITING_SYSTEM_PROMPT_MODE_IMPROVE_ABSTRACT = """
You are an expert scientist who excels in writing papers. Your task is to rewrite a given paper abstract to sound better and more human-like, using the full text of the paper to inform your rewriting. The abstract and text will be given in the next message. Please only return the rewritten text, do not return any intructions or anything like that. Only return the rewritten text. Make sure to make at least some edits or changes to the text.
""".strip()


def run_text_with_system_prompt(system_prompt: str, text: str) -> str:
    response = CLIENT.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=CONFIG.AI_ENDPOINT.MODEL_NAME,
    )
    assert response.choices and len(response.choices) > 0
    content = response.choices[0].message.content
    assert content is not None
    return content


def rewrite_abstract_only(abstract: str, _text: str) -> str:
    return run_text_with_system_prompt(
        REWRITING_SYSTEM_PROMPT_MODE_ABSTRACT_ONLY, abstract
    )


def write_new_abstract(_abstract: str, text: str) -> str:
    return run_text_with_system_prompt(REWRITING_SYSTEM_PROMPT_MODE_NEW_ABSTRACT, text)


def improve_abstract_text(abstract: str, text: str) -> str:
    return run_text_with_system_prompt(
        REWRITING_SYSTEM_PROMPT_MODE_IMPROVE_ABSTRACT,
        f"# Abstract:\n{abstract}\n\n# Full Text:\n{text}",
    )


mode_function_map: dict[str, Callable[[str, str], str]] = {
    "rewriteabstract": rewrite_abstract_only,
    "newabstract": write_new_abstract,
    "improveabstract": improve_abstract_text,
}

assert __name__ == "__main__", "Only run as script"

parser = ArgumentParser(description="Process paper abstracts through AI for rewriting.")
parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
parser.add_argument(
    "--mode", type=str, required=True, choices=list(mode_function_map.keys())
)
args = parser.parse_args()

with open(cast(str, args.input), "r") as f:
    data = cast(dict[str, str], json.load(f))

abstract_text = data["abstract"]
paper_text = data["text"]
paper_id = data["id"]
domain = data["domain"]

rewriter_fn = mode_function_map[cast(str, args.mode)]

new_abstract = rewriter_fn(abstract_text, paper_text)

new_data = {"abstract": new_abstract, "id": paper_id, "domain": domain}

with open(cast(str, args.output), "w") as f:
    json.dump(new_data, f, indent=2)
