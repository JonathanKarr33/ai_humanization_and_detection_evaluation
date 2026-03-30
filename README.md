# AI Humanization and Detection Evaluation

This project scrapes academic papers from OpenAlex, processes them through PANGRAM AI detection, and analyzes the results.

## Step 0: Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

Create a `.env` file in the project root with the following variables:

```
PANGRAM_API=your_pangram_api_key_here
GPT_ZERO_API_KEY=your_gptzero_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
EMAIL=your_email@example.com
UNDETECTABLE_USER_ID=your_user_id
UNDETECTABLE_API_KEY=your_api_key
```

- `PANGRAM_API`: Your PANGRAM API key for AI detection
- `GPT_ZERO_API_KEY`: Your GPTZero API key for AI detection
- `OPENROUTER_API_KEY`: Your OpenRouter key for LLM-assisted AI detection (used with `openai/gpt-5-nano`)
- `EMAIL`: Your email address (used for OpenAlex API requests)

`.env` formatting tips:
- Do not wrap values in quotes.
- Do not add spaces around `=`.
- Keep one key/value per line.

## Step 1: Paper Scrape

Scrape papers from OpenAlex where the domain is the #1 concept. Collects 100 papers per domain (political_science, theology, computer_science, chemistry).

```bash
python src/paper_scrape.py
```

This will:
- Fetch papers from OpenAlex API
- Filter for papers where the domain is the #1 concept
- Download PDFs and extract text
- Save metadata to `papers/{collection}/metadata_{collection}.jsonl` (default collection: `2020_back`)
- Save abstracts to `papers/{collection}/{domain}/abstracts/{paper_id}.txt`

**Note**: The script will automatically resume from previous runs and only collect the remaining papers needed to reach 100 per domain.

## Step 2: PANGRAM Abstracts

Process all paper abstracts through PANGRAM API for AI detection.

If you scraped into a non-default collection folder, pass `--collection` to match:

```bash
python src/paper_scrape.py --collection 2020_back
```

To scrape a specific publication date range (inclusive), use:

```bash
python src/paper_scrape.py --collection 2025_to_2023 --from-date 2023-01-01 --to-date 2025-12-31
```

## Visualize abstract coverage

Create a summary figure (counts by domain + by month) for any collection:

```bash
python src/visualize_abstracts.py --collection 2020_back --min-words 25
```

This writes to:

- `results/figures/{collection}/abstract_counts_min{min_words}.png`

```bash
python src/pangram_abstracts.py
```

Options:
- `--test`: Test mode - process only first 10 abstracts
- `--limit N`: Process only first N abstracts (use with `--test`)

This will:
- Read abstracts from `papers/{domain}/abstracts/`
- Send each abstract to PANGRAM API
- Save results to `pangram_abstracts_results.json`
- Skip abstracts already processed (resumable)

The output JSON includes:
- `paper_id`: Paper identifier (e.g., "W1580878179")
- `domain`: Domain name (e.g., "political_science")
- `ai_likelihood`: AI detection score (0-1)
- `prediction`: Text prediction (e.g., "Unlikely AI", "Possibly AI")
- `llm_prediction`: Breakdown by model (GPT35, GPT4, CLAUDE, etc.)
- All other PANGRAM response fields

## Step 3: Analyze Results

Analyze PANGRAM results to see distribution of AI likelihood scores.

```bash
python src/analyze_pangram_results.py
```

This will display:
- Overall distribution by 0.05 increments (0.00-0.05, 0.05-0.10, etc.)
- Distribution broken down by domain
- Summary statistics (min, max, mean, median)
- Key ranges summary (Very Unlikely AI, Unlikely AI, Uncertain, Likely AI)

## Step 4: Humanize Text with Undetectable.AI

This step runs Undetectable.AI over the paper abstracts (original + refined/new/polished variants) and saves both the original and humanized text plus full API metadata.

Naming note (important for future plots/graphs):
- We use **refine** where older code/data may say **improve**
- We use **polish** where older code/data may say **rewrite/rewritten**

```bash
python src/humanization_undetectable.py --collection 2025_back_2023
```

This will:
- Read original abstracts from `papers/{collection}/{domain}/paper_jsons/`
- Read refined/new/polished abstracts from `ai_improvement/{collection}/{domain}/*_abstracts/`
- For each abstract, call the Undetectable.AI humanization API (default: model v11, Doctorate readability, Article purpose, Balanced strength)
- Write one JSON per paper + variant to:
  - `humanization/{collection}/{domain}/{variant}/{paper_id}.json`

Each JSON includes:
- `paper_id`, `domain`, `variant`, `humanizer` (currently `"undetectable"`)
- `original_abstract`, `humanized_abstract`
- `undetectable.params`: settings used for the call
- `undetectable.document`: full response from Undetectable `/document`

## Step 5: Run AI Detection on Humanized Text

To evaluate AI-detection scores on the humanized abstracts, run:

```bash
python src/humanization_ai_detection.py --detector pangram --collection 2025_back_2023
python src/humanization_ai_detection.py --detector gptzero --collection 2025_back_2023
python src/humanization_ai_detection.py --detector llm_assisted --collection 2025_back_2023
```

This will:
- Read humanized abstracts from `humanization/{collection}/{domain}/{variant}/{paper_id}.json`
- Send each `humanized_abstract` to the selected detector API (`pangram`, `gptzero`, or `llm_assisted`)
- Save one JSON per paper + variant to:
- `humanization_results/{collection}/{domain}/{variant}_{detector}_results/{paper_id}.json`

Each JSON includes:
- `paper_id`, `domain`, `variant`
- `text`: the humanized abstract sent to the detector
- All fields returned by the selected detector API

For `llm_assisted`:
- Endpoint: `https://openrouter.ai/api/v1`
- Model: `openai/gpt-5-nano`
- Output includes `ai_probability` (0-1), `explanation`, and `model_name`

## Project Structure

```
.
├── papers/
│   ├── {collection}/
│   │   ├── metadata_{collection}.jsonl  # Paper metadata
│   │   ├── {domain}/
│   │   │   ├── abstracts/          # Abstract text files
│   │   │   ├── pdfs/               # PDF files
│   │   │   └── text/               # Extracted text files
├── ai_improvement/                 # Improved/new/rewritten abstracts by domain/variant
├── ai_improvement_results/         # Detector results on improved/new/rewritten (original pipeline)
├── humanization/
│   ├── {collection}/{domain}/{variant}/{paper_id}.json
│   │   # Original + humanized abstracts + Undetectable metadata
├── humanization_results/
│   ├── {collection}/{domain}/{variant}_{detector}_results/{paper_id}.json
│   │   # Detector results (PANGRAM/GPTZero/LLM-assisted) on humanized abstracts
├── pangram_abstracts_results.json  # PANGRAM detection results on original abstracts
├── src/
│   ├── paper_scrape.py             # Step 1: Scrape papers
│   ├── pangram_abstracts.py        # Step 2: Process original abstracts with PANGRAM
│   ├── analyze_pangram_results.py  # Step 3: Analyze original PANGRAM results
│   ├── humanization_undetectable.py# Step 4: Humanize abstracts with Undetectable
│   └── humanization_ai_detection.py# Step 5: Run detector on humanized abstracts
└── .env                            # Environment variables (create this)
```

## Requirements

- Python 3.9+
- Virtual environment
- PANGRAM API key
- GPTZero API key
- OpenRouter API key (for `llm_assisted`)
- Undetectable.AI API key
- Internet connection (for API calls)
