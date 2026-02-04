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


class MultiSourceOneTargetRecipeTemplate(MakefileRecipeTemplate):
    def __init__(self, target: str, commands: list[str]):
        super().__init__()
        self._target: str = target
        self._commands: list[str] = commands

    @override
    def get_commands(self) -> list[str]:
        return self._commands

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


def dir_to_dir_processor_recipes(
    srcs: list[str],
    new_dir: str,
    commands: list[str],
    common_name: str,
    processor_files: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    ind_recs, original_tgts = simple_makefile_recipe_template_factory(
        parent_dir_name_replacer(new_dir),
        [
            f"mkdir -p ./workarea/{new_dir}",
            *commands,
        ],
        persistent_srcs=processor_files,
    ).generate_recipes_for_single_sources(srcs, persistent_srcs=processor_files)
    common_runner_rec, _ = EmptyMultiSourceRecipeTemplate(common_name).generate_recipes(
        original_tgts
    )
    all_recs = ind_recs + common_runner_rec
    return all_recs, original_tgts


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


### ABSTRACT REWRITING WITH AI
# Rewrite with abstract only
all_rewrite_abstract_recs, rewrite_abstract_tgts = dir_to_dir_processor_recipes(
    srcs=paper_jsons,
    new_dir="rewritten_abstracts",
    commands=[
        "$(ENVPYTHON) ./src/rewrite_abstract.py --input $< --output $@ --mode rewriteabstract"
    ],
    processor_files=["./src/rewrite_abstract.py"],
    common_name="rewritten_abstracts",
)
all_recs.extend(all_rewrite_abstract_recs)
all_real_tgts.extend(rewrite_abstract_tgts)
all_recs.append("\n\n")

# Generate new abstract from text
all_new_abstract_recs, new_abstract_tgts = dir_to_dir_processor_recipes(
    srcs=paper_jsons,
    new_dir="new_abstracts",
    commands=[
        "$(ENVPYTHON) ./src/rewrite_abstract.py --input $< --output $@ --mode newabstract"
    ],
    processor_files=["./src/rewrite_abstract.py"],
    common_name="new_abstracts",
)
all_recs.extend(all_new_abstract_recs)
all_real_tgts.extend(new_abstract_tgts)
all_recs.append("\n\n")

# Improve abstract based on old abstract and text
all_improved_abstract_recs, improved_abstract_tgts = dir_to_dir_processor_recipes(
    srcs=paper_jsons,
    new_dir="improved_abstracts",
    commands=[
        "$(ENVPYTHON) ./src/rewrite_abstract.py --input $< --output $@ --mode improveabstract"
    ],
    processor_files=["./src/rewrite_abstract.py"],
    common_name="improved_abstracts",
)
all_recs.extend(all_improved_abstract_recs)
all_real_tgts.extend(improved_abstract_tgts)
all_recs.append("\n\n")

all_abstracts_recs, _ = EmptyMultiSourceRecipeTemplate("abstracts").generate_recipes(
    ["rewritten_abstracts", "new_abstracts", "improved_abstracts"]
)
all_recs.extend(all_abstracts_recs)
all_recs.append("\n\n")

### Pangram AI detection
# Original abstracts thru pangram
all_pangram_original_recs, pangram_original_tgts = dir_to_dir_processor_recipes(
    srcs=paper_jsons,
    new_dir="original_pangram_results",
    commands=["$(ENVPYTHON) ./src/pangram_abstract.py --input $< --output $@"],
    processor_files=["./src/pangram_abstract.py"],
    common_name="original_pangram",
)
all_recs.extend(all_pangram_original_recs)
all_real_tgts.extend(pangram_original_tgts)
all_recs.append("\n\n")

# Rewritten abstracts thru pangram
all_pangram_rewritten_recs, pangram_rewritten_tgts = dir_to_dir_processor_recipes(
    srcs=rewrite_abstract_tgts,
    new_dir="rewritten_pangram_results",
    commands=["$(ENVPYTHON) ./src/pangram_abstract.py --input $< --output $@"],
    processor_files=["./src/pangram_abstract.py"],
    common_name="rewritten_pangram",
)
all_recs.extend(all_pangram_rewritten_recs)
all_real_tgts.extend(pangram_rewritten_tgts)
all_recs.append("\n\n")

# New abstracts thru pangram
all_pangram_new_recs, pangram_new_tgts = dir_to_dir_processor_recipes(
    srcs=new_abstract_tgts,
    new_dir="new_pangram_results",
    commands=["$(ENVPYTHON) ./src/pangram_abstract.py --input $< --output $@"],
    processor_files=["./src/pangram_abstract.py"],
    common_name="new_pangram",
)
all_recs.extend(all_pangram_new_recs)
all_real_tgts.extend(pangram_new_tgts)
all_recs.append("\n\n")

# Improved abstracts thru pangram
all_pangram_improved_recs, pangram_improved_tgts = dir_to_dir_processor_recipes(
    srcs=improved_abstract_tgts,
    new_dir="improved_pangram_results",
    commands=["$(ENVPYTHON) ./src/pangram_abstract.py --input $< --output $@"],
    processor_files=["./src/pangram_abstract.py"],
    common_name="improved_pangram",
)
all_recs.extend(all_pangram_improved_recs)
all_real_tgts.extend(pangram_improved_tgts)
all_recs.append("\n\n")

all_pangram_recs, _ = EmptyMultiSourceRecipeTemplate("pangram").generate_recipes(
    ["original_pangram", "rewritten_pangram", "new_pangram", "improved_pangram"]
)
all_recs.extend(all_pangram_recs)
all_recs.append("\n\n")

all_pangram_tgts = (
    pangram_original_tgts
    + pangram_rewritten_tgts
    + pangram_new_tgts
    + pangram_improved_tgts
)
results_pangram_csv_rec, results_pangram_csv_tgt = MultiSourceOneTargetRecipeTemplate(
    "./workarea/results_pangram.csv",
    ["$(ENVPYTHON) ./src/pangram_to_csv.py --input $^ --output $@"],
).generate_recipes(all_pangram_tgts + ["./src/pangram_to_csv.py"])
all_recs.extend(results_pangram_csv_rec)
all_real_tgts.extend(results_pangram_csv_tgt)
result_pangram_pseudo_rec, _ = EmptyMultiSourceRecipeTemplate(
    "results_pangram"
).generate_recipes(results_pangram_csv_tgt)
all_recs.extend(result_pangram_pseudo_rec)
all_recs.append("\n\n")


### GPTZERO AI detection
# Original abstracts thru pangram
all_gptzero_original_recs, gptzero_original_tgts = dir_to_dir_processor_recipes(
    srcs=paper_jsons,
    new_dir="original_gptzero_results",
    commands=["$(ENVPYTHON) ./src/gptzero_abstract.py --input $< --output $@"],
    processor_files=["./src/gptzero_abstract.py"],
    common_name="original_gptzero",
)
all_recs.extend(all_gptzero_original_recs)
all_real_tgts.extend(gptzero_original_tgts)
all_recs.append("\n\n")

# Rewritten abstracts thru gptzero
all_gptzero_rewritten_recs, gptzero_rewritten_tgts = dir_to_dir_processor_recipes(
    srcs=rewrite_abstract_tgts,
    new_dir="rewritten_gptzero_results",
    commands=["$(ENVPYTHON) ./src/gptzero_abstract.py --input $< --output $@"],
    processor_files=["./src/gptzero_abstract.py"],
    common_name="rewritten_gptzero",
)
all_recs.extend(all_gptzero_rewritten_recs)
all_real_tgts.extend(gptzero_rewritten_tgts)
all_recs.append("\n\n")

# New abstracts thru gptzero
all_gptzero_new_recs, gptzero_new_tgts = dir_to_dir_processor_recipes(
    srcs=new_abstract_tgts,
    new_dir="new_gptzero_results",
    commands=["$(ENVPYTHON) ./src/gptzero_abstract.py --input $< --output $@"],
    processor_files=["./src/gptzero_abstract.py"],
    common_name="new_gptzero",
)
all_recs.extend(all_gptzero_new_recs)
all_real_tgts.extend(gptzero_new_tgts)
all_recs.append("\n\n")

# Improved abstracts thru gptzero
all_gptzero_improved_recs, gptzero_improved_tgts = dir_to_dir_processor_recipes(
    srcs=improved_abstract_tgts,
    new_dir="improved_gptzero_results",
    commands=["$(ENVPYTHON) ./src/gptzero_abstract.py --input $< --output $@"],
    processor_files=["./src/gptzero_abstract.py"],
    common_name="improved_gptzero",
)
all_recs.extend(all_gptzero_improved_recs)
all_real_tgts.extend(gptzero_improved_tgts)
all_recs.append("\n\n")

all_gptzero_recs, _ = EmptyMultiSourceRecipeTemplate("gptzero").generate_recipes(
    ["original_gptzero", "rewritten_gptzero", "new_gptzero", "improved_gptzero"]
)
all_recs.extend(all_gptzero_recs)
all_recs.append("\n\n")

all_gptzero_tgts = (
    gptzero_original_tgts
    + gptzero_rewritten_tgts
    + gptzero_new_tgts
    + gptzero_improved_tgts
)
results_gptzero_csv_rec, results_gptzero_csv_tgt = MultiSourceOneTargetRecipeTemplate(
    "./workarea/results_gptzero.csv",
    ["$(ENVPYTHON) ./src/gptzero_to_csv.py --input $^ --output $@"],
).generate_recipes(all_gptzero_tgts + ["./src/gptzero_to_csv.py"])
all_recs.extend(results_gptzero_csv_rec)
all_real_tgts.extend(results_gptzero_csv_tgt)
result_gptzero_pseudo_rec, _ = EmptyMultiSourceRecipeTemplate(
    "results_gptzero"
).generate_recipes(results_gptzero_csv_tgt)
all_recs.extend(result_gptzero_pseudo_rec)
all_recs.append("\n\n")

### LLM-Based AI detection
# Original abstracts thru llm_aid
all_llm_aid_original_recs, llm_aid_original_tgts = dir_to_dir_processor_recipes(
    srcs=paper_jsons,
    new_dir="original_llm_aid_results",
    commands=["$(ENVPYTHON) ./src/llm_aid_abstract.py --input $< --output $@"],
    processor_files=["./src/llm_aid_abstract.py"],
    common_name="original_llm_aid",
)
all_recs.extend(all_llm_aid_original_recs)
all_real_tgts.extend(llm_aid_original_tgts)
all_recs.append("\n\n")

# Rewritten abstracts thru llm_aid
all_llm_aid_rewritten_recs, llm_aid_rewritten_tgts = dir_to_dir_processor_recipes(
    srcs=rewrite_abstract_tgts,
    new_dir="rewritten_llm_aid_results",
    commands=["$(ENVPYTHON) ./src/llm_aid_abstract.py --input $< --output $@"],
    processor_files=["./src/llm_aid_abstract.py"],
    common_name="rewritten_llm_aid",
)
all_recs.extend(all_llm_aid_rewritten_recs)
all_real_tgts.extend(llm_aid_rewritten_tgts)
all_recs.append("\n\n")

# New abstracts thru llm_aid
all_llm_aid_new_recs, llm_aid_new_tgts = dir_to_dir_processor_recipes(
    srcs=new_abstract_tgts,
    new_dir="new_llm_aid_results",
    commands=["$(ENVPYTHON) ./src/llm_aid_abstract.py --input $< --output $@"],
    processor_files=["./src/llm_aid_abstract.py"],
    common_name="new_llm_aid",
)
all_recs.extend(all_llm_aid_new_recs)
all_real_tgts.extend(llm_aid_new_tgts)
all_recs.append("\n\n")

# Improved abstracts thru llm_aid
all_llm_aid_improved_recs, llm_aid_improved_tgts = dir_to_dir_processor_recipes(
    srcs=improved_abstract_tgts,
    new_dir="improved_llm_aid_results",
    commands=["$(ENVPYTHON) ./src/llm_aid_abstract.py --input $< --output $@"],
    processor_files=["./src/llm_aid_abstract.py"],
    common_name="improved_llm_aid",
)
all_recs.extend(all_llm_aid_improved_recs)
all_real_tgts.extend(llm_aid_improved_tgts)
all_recs.append("\n\n")

all_llm_aid_recs, _ = EmptyMultiSourceRecipeTemplate("llm_aid").generate_recipes(
    ["original_llm_aid", "rewritten_llm_aid", "new_llm_aid", "improved_llm_aid"]
)
all_recs.extend(all_llm_aid_recs)
all_recs.append("\n\n")

all_llm_aid_tgts = (
    llm_aid_original_tgts
    + llm_aid_rewritten_tgts
    + llm_aid_new_tgts
    + llm_aid_improved_tgts
)
results_llm_aid_csv_rec, results_llm_aid_csv_tgt = MultiSourceOneTargetRecipeTemplate(
    "./workarea/results_llm_aid.csv",
    ["$(ENVPYTHON) ./src/llm_aid_to_csv.py --input $^ --output $@"],
).generate_recipes(all_llm_aid_tgts + ["./src/llm_aid_to_csv.py"])
all_recs.extend(results_llm_aid_csv_rec)
all_real_tgts.extend(results_llm_aid_csv_tgt)
result_llm_aid_pseudo_rec, _ = EmptyMultiSourceRecipeTemplate(
    "results_llm_aid"
).generate_recipes(results_llm_aid_csv_tgt)
all_recs.extend(result_llm_aid_pseudo_rec)
all_recs.append("\n\n")

all_results_recs, _ = EmptyMultiSourceRecipeTemplate("results").generate_recipes(
    ["results_pangram", "results_gptzero", "results_llm_aid"]
)
all_recs.extend(all_results_recs)
all_recs.append("\n\n")

all_final_csv_tgts = (
    results_pangram_csv_tgt + results_gptzero_csv_tgt + results_llm_aid_csv_tgt
)
results_rec, results_tgt = MultiSourceOneTargetRecipeTemplate(
    "./workarea/results.csv",
    ["$(ENVPYTHON) ./src/merge_results.py --input $^ --output $@"],
).generate_recipes(all_final_csv_tgts + ["./src/merge_results.py"])
all_recs.extend(results_rec)
all_real_tgts.extend(results_tgt)
results_pseudo_rec, _ = EmptyMultiSourceRecipeTemplate("results").generate_recipes(
    results_tgt
)
all_recs.extend(results_pseudo_rec)
all_recs.append("\n\n")

# Mark all real files as precious so they're not deleted
precious_recs, _ = EmptyMultiSourceRecipeTemplate(".PRECIOUS").generate_recipes(
    all_real_tgts
)
all_recs.extend(precious_recs)

# Write all to the output file
all_recs_str = "\n".join(all_recs)
with open(cast(str, args.output), "w", encoding="utf-8") as outf:
    print(all_recs_str, file=outf)
