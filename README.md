# AI Humanization and Detection Evaluation

This project scrapes academic papers from OpenAlex, processes them through PANGRAM AI detection, and analyzes the results.

## Step 0: Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

Create a `.env` file in the project root with the following variables:

```
PANGRAM_API=your_pangram_api_key_here
EMAIL=your_email@example.com
```

- `PANGRAM_API`: Your PANGRAM API key for AI detection
- `EMAIL`: Your email address (used for OpenAlex API requests)

## Step 1: Paper Scrape

Scrape papers from OpenAlex where the domain is the #1 concept. Collects 100 papers per domain (political_science, theology, computer_science, chemistry).

```bash
python src/paper_scrape.py
```

This will:
- Fetch papers from OpenAlex API
- Filter for papers where the domain is the #1 concept
- Download PDFs and extract text
- Save metadata to `papers/metadata.jsonl`
- Save abstracts to `papers/{domain}/abstracts/{paper_id}.txt`

**Note**: The script will automatically resume from previous runs and only collect the remaining papers needed to reach 100 per domain.

## Step 2: PANGRAM Abstracts

Process all paper abstracts through PANGRAM API for AI detection.

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

## Project Structure

```
.
├── papers/
│   ├── metadata.jsonl              # Paper metadata
│   ├── {domain}/
│   │   ├── abstracts/              # Abstract text files
│   │   ├── pdfs/                   # PDF files
│   │   └── text/                   # Extracted text files
├── pangram_abstracts_results.json  # PANGRAM detection results
├── src/
│   ├── paper_scrape.py             # Step 1: Scrape papers
│   ├── pangram_abstracts.py        # Step 2: Process with PANGRAM
│   └── analyze_pangram_results.py  # Step 3: Analyze results
└── .env                            # Environment variables (create this)
```

## Requirements

- Python 3.9+
- Virtual environment
- PANGRAM API key
- Internet connection (for API calls)
