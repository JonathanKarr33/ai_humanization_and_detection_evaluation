# AI Humanization and Detection Evaluation

Controlled evaluation of commercial AI detectors on **published English abstracts** from OpenAlex. The pipeline samples four domains across two time windows (2013–2015 vs. 2023–2025), generates three LLM rewrite conditions per paper, scores text with Pangram, GPTZero, and an LLM-assisted baseline, optionally humanizes outputs with Undetectable.AI, and produces reproducible statistics and figures for error-analysis.

**Primary collections:** `2015_back_2013`, `2025_back_2023`  
**Domains:** `chemistry`, `computer_science`, `political_science`, `theology`

## Step 0: Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages: `pandas`, `matplotlib`, `seaborn`, `pangram-sdk`, `openai`, `requests`, `python-dotenv`, `nltk`.

### 3. Create `.env` file

Create a `.env` file in the project root:

```
PANGRAM_API=your_pangram_api_key_here
GPT_ZERO_API_KEY=your_gptzero_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
EMAIL=your_email@example.com
UNDETECTABLE_USER_ID=your_user_id
UNDETECTABLE_API_KEY=your_api_key
```

| Variable | Used for |
|----------|----------|
| `PANGRAM_API` | Pangram AI detection |
| `GPT_ZERO_API_KEY` | GPTZero API |
| `OPENROUTER_API_KEY` | LLM-assisted detector (`openai/gpt-5-nano`) and LLM abstract rewrites |
| `EMAIL` | OpenAlex API requests |
| `UNDETECTABLE_*` | Undetectable.AI humanization |

Formatting: one `KEY=value` per line, no quotes, no spaces around `=`.

## Rewrite conditions and on-disk names

Analysis and figures use these **display labels**. On disk, variant folders still use legacy directory names:

| Display label | Directory / `variant_raw` | Description |
|---------------|---------------------------|-------------|
| **original** | `original` | Unmodified abstract from OpenAlex |
| **refine (abstract only)** | `rewritten` / `rewritten_abstracts` | LLM rewrites abstract text only |
| **refine (abstract + paper)** | `improved` / `improved_abstracts` | LLM rewrites abstract using full paper |
| **new (article only)** | `new` / `new_abstracts` | LLM writes new abstract from full paper |

`src/paper_stats.py` and `src/figures.py` map raw names to display labels automatically.

## Step 1: Scrape papers (OpenAlex)

Collect abstracts and full text for each domain × collection:

```bash
python src/paper_scrape.py --collection 2015_back_2013 --from-date 2013-01-01 --to-date 2015-12-31
python src/paper_scrape.py --collection 2025_back_2023 --from-date 2023-01-01 --to-date 2025-12-31
```

Writes:

- `papers/{collection}/metadata_{collection}.jsonl`
- `papers/{collection}/{domain}/abstracts/{paper_id}.txt`
- `papers/{collection}/{domain}/paper_jsons/{paper_id}.json` (abstract + metadata for downstream steps)
- PDFs and extracted text under `pdfs/`, `text/`

The scraper resumes until ~100 papers per domain (before quality filtering).

### Abstract coverage figure

```bash
python src/figures.py abstracts --collection 2025_back_2023 --min-words 25
# or: python src/visualize_abstracts.py --collection 2025_back_2023 --min-words 25
```

Output: `results/figures/{collection}/abstract_counts_min{min_words}.png`

## Step 2: LLM abstract rewrites

Rewritten abstracts live under:

```
ai_improvement/{collection}/{domain}/
  improved_abstracts/{paper_id}.json
  new_abstracts/{paper_id}.json
  rewritten_abstracts/{paper_id}.json
```

Each JSON should include at least `id` (or paper id), `domain`, and `abstract` (rewrite text).

## Step 3: Detector scoring (pre-humanization)

Detector JSONs for **original + all rewrites** are stored under:

```
ai_improvement_results/{collection}/{domain}/
  {original|rewritten|improved|new}_{pangram|gptzero|llm_aid}_results/{paper_id}.json
```

Each file should include a detector score (`ai_likelihood` or `fraction_ai` for Pangram; `ai` for GPTZero; `ai_probability` for LLM-assisted) and ideally `text`, `paper_id`, and `domain`.

Use the same detector APIs as Step 5 (`humanization_ai_detection.py`) against the appropriate source text paths, or your own batch job that mirrors that layout.

## Step 4: Humanize with Undetectable.AI

```bash
python src/humanization_undetectable.py --collection 2025_back_2023
python src/humanization_undetectable.py --collection 2015_back_2013
```

Reads:

- Originals: `papers/{collection}/{domain}/paper_jsons/`
- Rewrites: `ai_improvement/{collection}/{domain}/*_abstracts/`

Writes:

```
humanization/{collection}/{domain}/{original|improved|new|rewritten}/{paper_id}.json
```

Each file contains `original_abstract`, `humanized_abstract`, and full Undetectable API metadata (`undetectable`).

Default settings: model v11, Balanced strength, Doctorate readability, Article purpose.

### Optional: stage humanization inputs only

```bash
python src/humanization.py --collection 2025_back_2023
```

Copies abstracts into `humanization/...` without calling Undetectable (useful for inspection).

## Step 5: Detector scoring (post-humanization)

```bash
python src/humanization_ai_detection.py --detector pangram --collection 2025_back_2023
python src/humanization_ai_detection.py --detector gptzero --collection 2025_back_2023
python src/humanization_ai_detection.py --detector llm_assisted --collection 2025_back_2023
```

Reads `humanization/{collection}/{domain}/{variant}/{paper_id}.json` and scores `humanized_abstract`.

Writes:

```
humanization_results/{collection}/{domain}/
  {variant}_{detector}_results/{paper_id}.json
```

Options: `--domains`, `--variants`, `--limit`, `--overwrite`.

## Step 6: Statistics and robustness

### Core statistical tests

Runs permutation tests, FDR-corrected feature correlations, domain-adjusted associations, length-partial correlations, and auto-selects one extreme pre/post example per collection:

```bash
python src/paper_stats.py
python src/paper_stats.py --collections 2015_back_2013 2025_back_2023 --n-perm 5000 --threshold 0.5
```

Outputs per collection under `results/statistics/{collection}/`:

| File | Contents |
|------|----------|
| `tests_humanization.json` | Matched pre/post humanization |
| `tests_domain_stem_vs_nonstem.json` | STEM vs non-STEM score contrasts |
| `tests_variant_effect.json` | Rewrite-condition ANOVA |
| `tests_detector_agreement.json` | Pangram vs GPTZero agreement |
| `tests_text_features.json` | Spearman + FDR text-feature tests |
| `example_case.json` | Auto-selected flip example |
| `robustness_threshold_sensitivity.csv` | FPR/FNR at τ ∈ {0.4, 0.5, 0.6} |
| `robustness_error_rates_ci.json` | Bootstrap CIs at τ = 0.5 |
| `robustness_paired_polish.json` | Paired original → refine (abstract only) |
| `robustness_coverage.json` | Pipeline completeness |
| `robustness_summary.md` | Combined robustness prose |
| `error_rates_all_detectors.csv` | Error rates by detector |
| `pr_curve_metrics.json` | AUC-PR per detector |
| `length_partial_correlations.csv` | Length-controlled ρ |

`paper_stats.py` also invokes `robustness_analysis.py` when available (writes PR figure path below).

### Humanization linguistic mechanisms

Paired pre/post analysis of `original_abstract` vs `humanized_abstract` (AWL, long-token ratio, type-token ratio, sentence length, etc.):

```bash
PYTHONPATH=src python3 src/humanization_linguistic_analysis.py
```

Writes `results/statistics/humanization_linguistics/` (`paired_feature_rows.csv`, `tests_humanization_features.json`, `summary.md`, `appendix_snippet.tex`).

### Robustness-only rerun

```bash
PYTHONPATH=src python src/robustness_analysis.py --collections 2015_back_2013 2025_back_2023
```

Adds/updates threshold tables, paired-shift stats, PR curves, and appends to `robustness_summary.md`.

## Step 7: Figures (`results/figures/`)

All plots are written under **`results/figures/{collection}/`**.

### Generate everything (recommended)

```bash
PYTHONPATH=src python src/figures.py all --collections 2015_back_2013 2025_back_2023
```

Runs abstract coverage (if configured), distribution grids, agreement scatters, and related outputs.

### Individual figure commands

```bash
# Pre-humanization 4×4 score distributions (domains × rewrite conditions)
PYTHONPATH=src python src/figures.py pangram --collections 2015_back_2013 2025_back_2023
PYTHONPATH=src python src/figures.py gptzero --collections 2015_back_2013 2025_back_2023
PYTHONPATH=src python src/figures.py llm-aid --collections 2015_back_2013 2025_back_2023

# Post-humanization distributions
PYTHONPATH=src python src/figures.py pangram-humanized --collections 2015_back_2013 2025_back_2023
PYTHONPATH=src python src/figures.py gptzero-humanized --collections 2015_back_2013 2025_back_2023

# Pangram vs GPTZero agreement (2×2 panels by condition)
PYTHONPATH=src python src/figures.py agreement --collections 2015_back_2013 2025_back_2023
PYTHONPATH=src python src/figures.py agreement-humanized --collections 2015_back_2013 2025_back_2023
```

### Standard output filenames

| File | Description |
|------|-------------|
| `pangram_distributions.png` | Pangram scores before humanization |
| `gptzero_distributions.png` | GPTZero scores before humanization |
| `llm_aid_distributions.png` | LLM-assisted scores before humanization |
| `post_huminization_ai_detection.png` | Pangram after humanization |
| `post_huminization_ai_detection_gptzero.png` | GPTZero after humanization |
| `pr_curves_pre.png` | Precision–recall (proxy labels, pre-humanization) |
| `pangram_vs_gptzero.png` | Agreement scatter, pre-humanization |
| `pangram_vs_gptzero_humanized.png` | Agreement scatter, post-humanization |
| `abstract_counts_min25.png` | Scrape coverage (from `abstracts` subcommand) |

With multiple `--collections`, files are placed in `results/figures/{collection}/` automatically.

### Alternate / legacy plot entry points

These wrap the same logic or older layouts:

```bash
python src/plot_pangram_grid.py --collections 2015_back_2013 2025_back_2023
python src/plot_detector_grids.py --collections 2015_back_2013 2025_back_2023
python src/plot_pangram_vs_gptzero_agreement.py --collections 2015_back_2013 2025_back_2023
python src/compare_pangram_distributions.py --collections 2015_back_2013 2025_back_2023
```

If matplotlib warns about cache permissions, set `export MPLCONFIGDIR=.mplconfig` (or another writable directory) before plotting.

## Legacy / exploratory scripts

Earlier workflow steps remain in `src/` for one-off use:

| Script | Purpose |
|--------|---------|
| `pangram_abstracts.py` | Batch Pangram on scraped abstracts only |
| `analyze_pangram_results.py` | Histogram summaries of early Pangram JSON |
| `pangram_test.py` | Ad-hoc Pangram API tests |
| `summarize_preai_pangram.py` | Domain × type summaries of `ai_improvement_results` |

## Project structure

```
.
├── papers/                         # OpenAlex scrape per collection/domain
├── ai_improvement/                 # LLM rewrite JSONs (*_abstracts/)
├── ai_improvement_results/         # Detector scores before humanization
├── humanization/                   # Original + humanized text (Undetectable)
├── humanization_results/           # Detector scores after humanization
├── results/
│   ├── figures/{collection}/       # All generated plots
│   └── statistics/{collection}/  # Tests, tables, example snippets
├── src/
│   ├── paper_scrape.py
│   ├── humanization_undetectable.py
│   ├── humanization_ai_detection.py
│   ├── humanization.py             # Stage humanization folders (no API)
│   ├── paper_stats.py              # Main analysis + triggers robustness
│   ├── robustness_analysis.py      # Threshold/PR/paired robustness
│   ├── figures.py                  # Unified figure generator
│   ├── plot_pangram_grid.py
│   ├── plot_detector_grids.py
│   ├── plot_pangram_vs_gptzero_agreement.py
│   └── visualize_abstracts.py
└── .env
```

## Requirements

- Python 3.9+
- API keys listed above
- Internet access for OpenAlex and detector APIs
