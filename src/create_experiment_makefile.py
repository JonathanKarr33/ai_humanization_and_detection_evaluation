import pathlib
from argparse import ArgumentParser
from collections.abc import Callable
from glob import glob
from typing import cast, override


def create_makefile_recipe_topline(srcs: list[str], tgt: str) -> str:
    srcs_str = " ".join([e.strip() for e in srcs if len(e.strip()) > 0])
    return f"{tgt.strip()}: {srcs_str.strip()}"


def create_makefile_recipe(srcs: list[str], tgt: str, commands: list[str]) -> str:
    topline = create_makefile_recipe_topline(srcs, tgt)
    if len(commands) == 0:
        return f"{topline}\n\n"
    commands = [f"\t{c.strip()}" for c in commands if len(c.strip()) > 0]
    commands_str = "\n".join(commands)
    rec = f"{topline}\n{commands_str}"
    return rec


def get_dir_name(p: str):
    _p = pathlib.Path(p)
    return _p.parent.name


class MakefileRecipeTemplate(object):
    def __init__(self):
        super().__init__()

    def get_commands(self) -> list[str]:
        return []

    def srcs_to_tgts(self, srcs: list[str]) -> list[str]:
        return ["default"]

    def generate_recipes(self, srcs: list[str]) -> tuple[list[str], list[str]]:
        tgts = self.srcs_to_tgts(srcs)
        cmds = self.get_commands()
        recs = [create_makefile_recipe(srcs, tgt, cmds) for tgt in tgts]
        return recs, tgts

    def generate_recipes_for_single_sources(
        self, srcs: list[str], persistent_srcs: list[str] | None = None
    ) -> tuple[list[str], list[str]]:
        tgts: list[str] = []
        recs: list[str] = []
        for s in srcs:
            _s = [s]
            if persistent_srcs is not None:
                _s.extend(persistent_srcs)
            c_recs, c_tgts = self.generate_recipes(_s)
            tgts.extend(c_tgts)
            recs.extend(c_recs)
        return recs, tgts


class EmptyMultiSourceRecipeTemplate(MakefileRecipeTemplate):
    def __init__(self, target: str):
        super().__init__()
        self._target: str = target

    @override
    def get_commands(self) -> list[str]:
        return []

    @override
    def srcs_to_tgts(self, srcs: list[str]) -> list[str]:
        return [self._target]


def simple_makefile_recipe_template_factory(
    name_transformer: Callable[[str], str],
    commands: list[str],
    persistent_srcs: list[str] | None = None,
) -> MakefileRecipeTemplate:
    class _ProxyTemplate(MakefileRecipeTemplate):
        def __init__(self):
            super().__init__()

        @override
        def get_commands(self) -> list[str]:
            return commands

        @override
        def srcs_to_tgts(self, srcs: list[str]) -> list[str]:
            if persistent_srcs is not None:
                srcs = sorted(list(set(srcs) - set(persistent_srcs)))
            assert len(srcs) == 1
            return [name_transformer(s) for s in srcs]

    return _ProxyTemplate()


def parent_dir_name_replacer(new_dir: str) -> Callable[[str], str]:
    def _t(p: str):
        parent = get_dir_name(p)
        return p.replace(parent, new_dir)

    return _t


assert __name__ == "__main__", "Always run this as script"

aprs = ArgumentParser()
aprs.add_argument("--papers", type=str, required=True, help="Path to the paper JSONS")
aprs.add_argument(
    "--output", type=str, required=True, help="Where to save the generated makefile"
)
args = aprs.parse_args()

paper_glob = f"{cast(str, args.papers)}/*.json"
paper_jsons = glob(paper_glob)

all_recs: list[str] = []
all_real_tgts: list[str] = []

# Original abstracts thru pangram
ORIGINAL_PANGRAM_OUTPUT_DIR = "original_pangram_results"
pangram_original_recs, pangram_original_tgts = simple_makefile_recipe_template_factory(
    parent_dir_name_replacer(ORIGINAL_PANGRAM_OUTPUT_DIR),
    [
        f"mkdir -p ./workarea/{ORIGINAL_PANGRAM_OUTPUT_DIR}",
        "$(ENVPYTHON) ./src/pangram_abstract.py --input $< --output $@",
    ],
    persistent_srcs=["./src/pangram_abstract.py"],
).generate_recipes_for_single_sources(
    paper_jsons, persistent_srcs=["./src/pangram_abstract.py"]
)
run_all_original_pangram_recs, _ = EmptyMultiSourceRecipeTemplate(
    "original_pangram"
).generate_recipes(pangram_original_tgts)
all_pangram_original_recs = pangram_original_recs + run_all_original_pangram_recs
all_recs.extend(all_pangram_original_recs)
all_recs.append("\n\n")
all_real_tgts.extend(pangram_original_tgts)


# Mark all JSONS as precious so they're not deleted
precious_recs, _ = EmptyMultiSourceRecipeTemplate(".PRECIOUS").generate_recipes(
    all_real_tgts
)
all_recs.extend(precious_recs)


# Write all to the output file
all_recs_str = "\n".join(all_recs)
with open(cast(str, args.output), "w", encoding="utf-8") as outf:
    print(all_recs_str, file=outf)
