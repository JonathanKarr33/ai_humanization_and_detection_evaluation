from __future__ import annotations

import os
import json
import time
import requests
from tqdm import tqdm
import fitz  # PyMuPDF
import re
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# ==========================
# CONFIG
# ==========================
OPENALEX_API = "https://api.openalex.org/works"
# Use EMAIL from .env, fallback to OPENALEX_EMAIL, then default
EMAIL = os.getenv("EMAIL") or os.getenv("OPENALEX_EMAIL", "jkarr@nd.edu")  # required by OpenAlex etiquette
PAPERS_ROOT_DIR = "papers"
DEFAULT_COLLECTION = "2020_back"
OUTPUT_DIR = os.path.join(PAPERS_ROOT_DIR, DEFAULT_COLLECTION)
META_FILE = os.path.join(OUTPUT_DIR, f"metadata_{DEFAULT_COLLECTION}.jsonl")
# Domain-specific folders: papers/{collection}/{domain}/pdfs/, text/, abstracts/

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# OpenAlex concept IDs
# Verified via OpenAlex API:
# - Political Science (C17744445): https://openalex.org/C17744445
# - Theology (C27206212): https://openalex.org/C27206212
# - Computer Science (C41008148): https://openalex.org/C41008148
# - Chemistry (C185592680): https://openalex.org/C185592680
CONCEPTS = {
    "political_science": "C17744445",  # Political science - verified ✓
    "theology": "C27206212",  # Theology - verified ✓
    "computer_science": "C41008148",  # Computer science
    "chemistry": "C185592680"  # Chemistry
}

MAX_PAPERS_PER_DOMAIN = 100  # Target number of papers to collect per domain
# Abstracts must be at least 25 words.
MIN_ABSTRACT_WORDS = 25
PER_PAGE = 200
YEAR_CUTOFF = 2020
# Optional inclusive publication date range (OpenAlex filter sugar):
# - from_publication_date:YYYY-MM-DD
# - to_publication_date:YYYY-MM-DD
DATE_FROM = None
DATE_TO = None
# If True, only record/count papers when a PDF download succeeds.
# This is useful for tightly-scoped date-range collections where you want guaranteed PDFs.
REQUIRE_PDF = False
MAX_WORKERS = 10
# How many candidate papers to download/process per batch.
# Bigger = fewer OpenAlex roundtrips; smaller = less wasted work when many PDFs fail.
PROCESS_BATCH_SIZE = 80
# Full-text extraction is expensive and not needed for "abstracts-only" workflows.
# We still download PDFs when required to recover abstracts, but we skip full-text extraction.
EXTRACT_FULLTEXT = False

# ==========================
# SETUP
# ==========================
# Create base output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Domain-specific directories created as needed


def configure_output(collection: str) -> None:
    """Configure output folder under papers/{collection}/..."""
    global OUTPUT_DIR, META_FILE
    OUTPUT_DIR = os.path.join(PAPERS_ROOT_DIR, collection)
    META_FILE = os.path.join(OUTPUT_DIR, f"metadata_{collection}.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def configure_date_range(date_from: Optional[str], date_to: Optional[str]) -> None:
    """Configure optional OpenAlex publication date range (inclusive)."""
    global DATE_FROM, DATE_TO
    DATE_FROM = date_from
    DATE_TO = date_to


def configure_require_pdf(require_pdf: bool) -> None:
    global REQUIRE_PDF
    REQUIRE_PDF = bool(require_pdf)

# Headers for OpenAlex API (simple, as per their documentation)
api_headers = {
    "User-Agent": f"PaperScraper ({EMAIL})",
    "Accept": "application/json"
}

# Reuse sessions for better performance (connection pooling).
openalex_session = requests.Session()
openalex_session.headers.update(api_headers)


def _candidate_pdf_urls_from_work(work: dict) -> list:
    """
    Return a prioritized list of URLs that might yield a PDF.
    OpenAlex often provides better links via best_oa_location than open_access.oa_url.
    """
    urls = []
    best = work.get("best_oa_location") or {}
    if isinstance(best, dict):
        for k in ("pdf_url", "url"):
            v = best.get(k)
            if v:
                urls.append(v)

    primary = work.get("primary_location") or {}
    if isinstance(primary, dict):
        for k in ("pdf_url", "landing_page_url"):
            v = primary.get(k)
            if v:
                urls.append(v)

    oa = work.get("open_access") or {}
    if isinstance(oa, dict):
        v = oa.get("oa_url")
        if v:
            urls.append(v)

    # Unique preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_openalex_work(openalex_id: str):
    """Fetch a full work record from OpenAlex (best_oa_location, etc.)."""
    if not openalex_id:
        return None
    work_id = openalex_id
    if openalex_id.startswith("https://openalex.org/"):
        work_id = openalex_id.replace("https://openalex.org/", "")
    url = f"{OPENALEX_API}/{work_id}"
    params = {"mailto": EMAIL}
    try:
        # Keep a small sleep to be polite; backfill may call this many times.
        time.sleep(0.12)
        r = openalex_session.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            r = openalex_session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# Browser-like headers for PDF downloads to avoid 403 errors
pdf_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# ==========================
# HELPERS
# ==========================
def abstract_word_count(text: str) -> int:
    if not text:
        return 0
    # Count "word-like" tokens.
    return len(re.findall(r"\b\w+\b", text))


def publication_month_from_date(publication_date: Optional[str]):
    """Return YYYY-MM from YYYY-MM-DD, else None."""
    if not publication_date or not isinstance(publication_date, str):
        return None
    m = re.match(r"^(\d{4}-\d{2})-\d{2}$", publication_date.strip())
    return m.group(1) if m else None


def extract_publication_date(work: dict):
    """Prefer OpenAlex `publication_date` if present."""
    d = work.get("publication_date")
    if isinstance(d, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return d
    return None


def fetch_openalex(concept_id):
    cursor = "*"
    page_count = 0
    # No page limit - we'll stop when we have enough papers

    while cursor:
        # Build OpenAlex filter string.
        filters = [
            f"concepts.id:{concept_id}",
            "open_access.is_oa:true",
            "language:en",
        ]
        # Prefer explicit date range if provided; otherwise fall back to legacy year cutoff.
        if DATE_FROM:
            filters.append(f"from_publication_date:{DATE_FROM}")
        if DATE_TO:
            filters.append(f"to_publication_date:{DATE_TO}")
        if not DATE_FROM and not DATE_TO:
            filters.append(f"publication_year:<{YEAR_CUTOFF}")

        filter_str = ",".join(filters)
        params = {
            "filter": filter_str,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "mailto": EMAIL,  # polite pool
        }

        try:
            r = openalex_session.get(OPENALEX_API, params=params, timeout=30)
            if r.status_code == 429:
                # Back off and retry this page (do not advance cursor).
                time.sleep(5)
                continue
            r.raise_for_status()
            
            # Check if response is actually JSON
            content_type = r.headers.get("content-type", "").lower()
            if "application/json" not in content_type:
                print(f"Warning: Expected JSON, got {content_type}")
                print(f"Response preview: {r.text[:200]}")
                break
            
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response preview: {r.text[:500]}")
                break
            
            if "results" not in data:
                print(f"Unexpected API response structure: {list(data.keys())}")
                break
                
            if not data.get("results"):
                # No more results
                break

            for item in data["results"]:
                yield item

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            page_count += 1
            # OpenAlex limit is 10 req/sec; this endpoint is one request per page.
            # Keep a small sleep to be polite without making large skips unbearably slow.
            time.sleep(0.12)
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            break
        except Exception as e:
            print(f"Unexpected error fetching from OpenAlex: {e}")
            break


def download_pdf(url, path):
    """Download PDF from URL and save to path. Returns True if successful."""
    try:
        # Allow redirects and use session for better handling
        session = requests.Session()
        session.headers.update(pdf_headers)
        
        r = session.get(url, timeout=30, allow_redirects=True, stream=True)
        
        # Check if we got a PDF
        content_type = r.headers.get("content-type", "").lower()
        # Some providers return PDFs as application/octet-stream; verify using the PDF magic header.
        first_chunk = None
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                first_chunk = chunk
                break

        if first_chunk is None:
            return False

        is_pdf_by_type = content_type.startswith("application/pdf")
        is_pdf_by_magic = first_chunk.lstrip().startswith(b"%PDF")

        if is_pdf_by_type or is_pdf_by_magic:
            with open(path, "wb") as f:
                f.write(first_chunk)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return os.path.exists(path) and os.path.getsize(path) > 0
        
        # If we got HTML, try to find a PDF link (common for DOI/landing pages)
        if content_type.startswith("text/html"):
            # Some sites block, but we'll try to extract PDF link from common patterns
            # For now, just skip - many require JavaScript or have complex auth
            return False
        else:
            # Unknown content type
            return False
            
    except requests.exceptions.HTTPError as e:
        # Don't print every 403 - too noisy. Only print if it's unexpected
        if e.response.status_code != 403:
            print(f"HTTP {e.response.status_code} downloading from {url}")
        return False
    except requests.RequestException as e:
        # Only print connection errors, not all request errors
        if "Connection" in str(type(e).__name__):
            print(f"Connection error for {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading PDF: {e}")
        return False


def extract_openalex_id(work):
    """Extract stable unique ID from OpenAlex work object.
    Returns format like 'W123456789' from 'https://openalex.org/W123456789'
    """
    openalex_id = work.get("id", "")
    if not openalex_id:
        return None
    
    # Extract the ID part (e.g., "W123456789" from "https://openalex.org/W123456789")
    match = re.search(r'/([W]\d+)$', openalex_id)
    if match:
        return match.group(1)
    
    # Fallback: use a hash of the full ID if format is unexpected
    import hashlib
    return hashlib.md5(openalex_id.encode()).hexdigest()[:12]


def reconstruct_abstract_from_inverted_index(inverted_index):
    """Reconstruct abstract text from OpenAlex abstract_inverted_index format.
    Returns the abstract as a string or None.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    
    try:
        # Create list of (position, word) tuples
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sort by position and join words
        word_positions.sort(key=lambda x: x[0])
        abstract = ' '.join(word for _, word in word_positions)
        return abstract if abstract.strip() else None
    except Exception:
        return None


def extract_abstract_from_pdf(pdf_path):
    """Extract abstract from PDF by looking at first few pages.
    Returns abstract text or None if not found.
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Check first 3 pages for abstract (most abstracts are on page 1-2)
        abstract_text = None
        abstract_patterns = [
            r'(?i)\babstract\b[:\s]*(.*?)(?=\n\s*(?:introduction|keywords?|background|1\.|introduction|method|result))',
            r'(?i)\babstract\b[:\s]*(.*?)(?=\n\n\n)',  # Stop at triple newline
            r'(?i)\babstract\b[:\s]*(.{100,2000})',  # Generic: 100-2000 chars after "Abstract"
        ]
        
        # Get text from first few pages
        text_to_search = ""
        for page_num in range(min(3, len(doc))):
            page_text = doc[page_num].get_text()
            if isinstance(page_text, str):
                text_to_search += page_text + "\n\n"
        
        doc.close()
        
        # Try to find abstract using patterns
        for pattern in abstract_patterns:
            match = re.search(pattern, text_to_search, re.DOTALL)
            if match:
                abstract_text = match.group(1).strip()
                # Clean up: remove extra whitespace, limit length
                abstract_text = re.sub(r'\s+', ' ', abstract_text)
                if 50 <= len(abstract_text) <= 2000:  # Reasonable abstract length
                    return abstract_text
        
        # Fallback: if we see "Abstract" heading, take next paragraph
        lines = text_to_search.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'(?i)^\s*abstract\s*$', line):
                # Take next few non-empty lines
                abstract_lines = []
                for j in range(i + 1, min(i + 20, len(lines))):
                    if lines[j].strip():
                        abstract_lines.append(lines[j].strip())
                    elif abstract_lines:  # Stop at first empty line after content
                        break
                if abstract_lines:
                    abstract_text = ' '.join(abstract_lines).strip()
                    if 50 <= len(abstract_text) <= 2000:
                        return abstract_text
        
        return None
        
    except Exception as e:
        # Silently fail - abstract extraction is optional
        return None


def grobid_extract(pdf_path):
    """Extract text from PDF using GROBID. Returns text or None if failed."""
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": f}
            r = requests.post(GROBID_URL, files=files, timeout=5)  # Short timeout since it's optional
        if r.status_code == 200:
            return r.text
        return None
    except requests.exceptions.ConnectionError:
        # GROBID not running - silently fall back to PyMuPDF
        return None
    except requests.RequestException:
        # Other GROBID errors - silently fall back
        return None
    except Exception:
        # Any other error - silently fall back
        return None


# ==========================
# MAIN PIPELINE
# ==========================
def test_api_connection():
    """Test if OpenAlex API is accessible."""
    try:
        test_params = {"per-page": 1}
        r = requests.get(OPENALEX_API, params=test_params, headers=api_headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "results" in data:
            print("✅ OpenAlex API connection successful")
            return True
        else:
            print(f"⚠️  API responded but unexpected format: {list(data.keys())}")
            return False
    except Exception as e:
        print(f"❌ OpenAlex API connection failed: {e}")
        print("   Please check your internet connection and try again.")
        return False


# Rate limiter for OpenAlex API (10 requests per second max)
class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = Lock()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                # Remove calls older than the period
                while self.calls and self.calls[0] < now - self.period:
                    self.calls.popleft()
                
                # Wait if we're at the limit
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        # Clean up again after waiting
                        now = time.time()
                        while self.calls and self.calls[0] < now - self.period:
                            self.calls.popleft()
                
                self.calls.append(time.time())
            
            return func(*args, **kwargs)
        return wrapper


# Create rate limiter: 9 requests per second (one below OpenAlex's 10 req/sec limit)
rate_limiter = RateLimiter(max_calls=9, period=1.0)


@rate_limiter
def get_paper_concepts(openalex_id: str, max_retries: int = 5):
    """Fetch paper from OpenAlex API and return its concepts list with retry logic."""
    # Extract work ID from openalex_id URL if it's a full URL
    work_id = openalex_id
    if openalex_id.startswith("https://openalex.org/"):
        work_id = openalex_id.replace("https://openalex.org/", "")
    
    url = f"{OPENALEX_API}/{work_id}"
    params = {"mailto": EMAIL}  # Join polite pool
    
    for attempt in range(max_retries):
        try:
            # Rate limiter ensures we don't exceed 10 req/sec
            response = requests.get(url, headers=api_headers, params=params, timeout=30)
            
            if response.status_code == 429:
                # Exponential backoff for rate limit errors
                wait_time = min(2 ** attempt, 60)  # Cap at 60 seconds
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed
                    return None
            
            response.raise_for_status()
            data = response.json()
            return data.get("concepts", [])
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)
                time.sleep(wait_time)
                continue
            # Don't print errors for rate limits - too noisy
            if "429" not in str(e):
                print(f"Error fetching {openalex_id}: {e}")
            return None
        except Exception as e:
            if "429" not in str(e):
                print(f"Error fetching {openalex_id}: {e}")
            return None
    
    return None


def is_domain_first_concept(concepts: list, domain: str) -> bool:
    """Check if the domain concept is ranked #1 (highest score)."""
    if not concepts:
        return False
    
    concept_id = CONCEPTS.get(domain)
    if not concept_id:
        return False
    
    # Concepts are sorted by relevance score (highest first)
    # Check if the first concept matches our domain
    if concepts and len(concepts) > 0:
        first_concept = concepts[0]
        first_concept_id = first_concept.get("id", "")
        if concept_id in first_concept_id or first_concept_id.endswith(concept_id):
            return True
    
    return False


# Note: All papers in metadata.jsonl are already filtered for #1 concept during scraping
# No need for additional filtering step or first/ directories


def process_paper_download(work_data):
    """Process a single paper: download PDF, extract text, return metadata."""
    work, domain, pid, pdf_url = work_data
    
    domain_dir = os.path.join(OUTPUT_DIR, domain)
    pdf_dir = os.path.join(domain_dir, "pdfs")
    text_dir = os.path.join(domain_dir, "text")
    abstract_dir = os.path.join(domain_dir, "abstracts")
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(abstract_dir, exist_ok=True)
    
    pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
    txt_path = os.path.join(text_dir, f"{pid}.txt")
    
    # Prefer OpenAlex-provided abstracts first to avoid unnecessary PDF downloads.
    # Many works include `abstract_inverted_index`; reconstructing it is fast.
    abstract = work.get("abstract")
    if not abstract:
        inverted_index = work.get("abstract_inverted_index")
        if inverted_index:
            abstract = reconstruct_abstract_from_inverted_index(inverted_index)

    def _ensure_pdf() -> bool:
        """Ensure a PDF exists on disk for this work (best-effort)."""
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            return True
        # Try the candidate passed in, then fall back to best OpenAlex locations.
        urls = []
        if pdf_url:
            urls.append(pdf_url)
        urls.extend(_candidate_pdf_urls_from_work(work))
        seen = set()
        for u in urls:
            if not u or u in seen:
                continue
            seen.add(u)
            if download_pdf(u, pdf_path):
                return True
        return os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0

    # If OpenAlex already provides a good abstract, keep it and save it.
    # If REQUIRE_PDF is enabled, only succeed if we can also download the PDF.
    if abstract and abstract.strip() and abstract_word_count(abstract) >= MIN_ABSTRACT_WORDS:
        if REQUIRE_PDF and (not _ensure_pdf()):
            return None
        publication_date = extract_publication_date(work)
        try:
            abstract_path = os.path.join(abstract_dir, f"{pid}.txt")
            with open(abstract_path, "w", encoding="utf-8") as f:
                f.write(abstract.strip())
        except Exception:
            pass

        return {
            "id": pid,
            "openalex_id": work.get("id"),
            "domain": domain,
            "title": work.get("display_name"),
            "year": work.get("publication_year"),
            "publication_date": publication_date,
            "publication_month": publication_month_from_date(publication_date),
            "doi": work.get("doi"),
            "abstract": abstract.strip(),
            "pdf_url": pdf_url,
        }

    # Check if already processed (for require-pdf mode, PDF must exist)
    pdf_exists = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
    if pdf_exists and os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        # Already processed
        abstract = work.get("abstract")
        if not abstract:
            inverted_index = work.get("abstract_inverted_index")
            if inverted_index:
                abstract = reconstruct_abstract_from_inverted_index(inverted_index)
        if not abstract:
            abstract = extract_abstract_from_pdf(pdf_path)
        if abstract and abstract.strip() and abstract_word_count(abstract) >= MIN_ABSTRACT_WORDS:
            if REQUIRE_PDF and not pdf_exists:
                return None
            publication_date = extract_publication_date(work)
            # Ensure abstract file exists and is up to date
            try:
                abstract_path = os.path.join(abstract_dir, f"{pid}.txt")
                with open(abstract_path, "w", encoding="utf-8") as f:
                    f.write(abstract.strip())
            except Exception:
                pass
            return {
                "id": pid,
                "openalex_id": work.get("id"),
                "domain": domain,
                "title": work.get("display_name"),
                "year": work.get("publication_year"),
                "publication_date": publication_date,
                "publication_month": publication_month_from_date(publication_date),
                "doi": work.get("doi"),
                "abstract": abstract,
                "pdf_url": pdf_url
            }
        return None
    
    # Download PDF
    if REQUIRE_PDF or (abstract is None):
        # If we need the PDF (require-pdf mode or to recover abstract), ensure it's downloaded.
        if not _ensure_pdf():
            return None
        pdf_exists = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
    
    # Ensure we have a valid abstract (prefer OpenAlex, fallback to PDF)
    if not abstract:
        if not pdf_exists:
            return None
        abstract = extract_abstract_from_pdf(pdf_path)
    if not abstract or not abstract.strip():
        return None
    if abstract_word_count(abstract) < MIN_ABSTRACT_WORDS:
        return None

    publication_date = extract_publication_date(work)
    
    # Save abstract
    try:
        abstract_path = os.path.join(abstract_dir, f"{pid}.txt")
        with open(abstract_path, "w", encoding="utf-8") as f:
            f.write(abstract.strip())
    except Exception:
        pass
    
    # Extract text (optional; many PDFs are image-only). Best-effort only.
    if EXTRACT_FULLTEXT and (not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0):
        tei_xml = grobid_extract(pdf_path)
        if tei_xml and tei_xml.strip():
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(tei_xml)
            except Exception:
                pass
        else:
            try:
                doc = fitz.open(pdf_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                text = "\n".join(text_parts)
                if text.strip():
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
            except Exception:
                pass
    
    return {
        "id": pid,
        "openalex_id": work.get("id"),
        "domain": domain,
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "publication_date": publication_date,
        "publication_month": publication_month_from_date(publication_date),
        "doi": work.get("doi"),
        "abstract": abstract,
        "pdf_url": pdf_url
    }


def scrape_papers():
    """Scrape papers from OpenAlex and save to metadata.jsonl."""
    if EMAIL == "your_email@example.com":
        print("Warning: Using placeholder email. Set EMAIL or OPENALEX_EMAIL in .env file.")
    
    # Test API connection first
    if not test_api_connection():
        print("\nExiting due to API connection issues.")
        return False
    
    def _has_pdf(domain: str, pid: str) -> bool:
        pdf_path = os.path.join(OUTPUT_DIR, domain, "pdfs", f"{pid}.pdf")
        return os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0

    def _is_valid_existing(paper: dict) -> bool:
        # Count only papers whose abstract meets the minimum word requirement (and pdf if required).
        abs_text = paper.get("abstract") or ""
        if abstract_word_count(abs_text) < MIN_ABSTRACT_WORDS:
            return False
        if REQUIRE_PDF:
            dom = paper.get("domain")
            pid = paper.get("id")
            if not dom or not pid:
                return False
            return _has_pdf(dom, pid)
        return True

    # Check existing papers in metadata.jsonl (dedup by paper id; count only valid abstracts)
    existing_papers: dict = {}
    if os.path.exists(META_FILE):
        print("Checking existing papers in metadata.jsonl...")
        with open(META_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        paper = json.loads(line)
                        paper_id = paper.get("id")
                        if paper_id:
                            existing_papers[paper_id] = paper
                    except:
                        pass

    # Compute valid counts by domain from the de-duplicated mapping
    existing_by_domain = {}
    for paper in existing_papers.values():
        domain = paper.get("domain")
        if domain and _is_valid_existing(paper):
            existing_by_domain[domain] = existing_by_domain.get(domain, 0) + 1

    if existing_by_domain:
        print("Found existing papers (counting only abstracts >= "
              f"{MIN_ABSTRACT_WORDS} words):")
        for domain, count in existing_by_domain.items():
            print(f"  {domain}: {count} papers")
        print(f"Total existing (unique ids): {len(existing_papers)} papers\n")
    
    # Process each domain separately to ensure we get 100 per domain
    print("Step 1: Collecting and processing papers where domain is #1 concept...")
    total_processed = sum(existing_by_domain.values())
    
    # Write mode - we'll append new papers as we process them
    with open(META_FILE, "a" if existing_papers else "w", encoding="utf-8") as meta_out:
        for domain, concept_id in CONCEPTS.items():
            # Check how many we already have for this domain
            existing_count = existing_by_domain.get(domain, 0)
            needed = max(0, MAX_PAPERS_PER_DOMAIN - existing_count)
            
            if existing_count >= MAX_PAPERS_PER_DOMAIN:
                print(f"\n📚 {domain}: Already have {existing_count}/{MAX_PAPERS_PER_DOMAIN} papers (from earlier run). Skipping.")
                continue
            
            print(
                f"\n📚 Processing {domain} papers (target: {MAX_PAPERS_PER_DOMAIN}, "
                f"have: {existing_count}, need: {needed}; abstracts must be >= {MIN_ABSTRACT_WORDS} words)..."
            )

            # Avoid reprocessing IDs that already meet the abstract-length requirement.
            # BUT allow reprocessing of IDs that exist but have too-short abstracts, so we can
            # potentially recover a longer abstract from the PDF and upgrade them to "valid".
            processed_ids = set(
                pid for pid, paper in existing_papers.items() if _is_valid_existing(paper)
            )
            domain_processed = existing_count

            print(f"  Processing {domain} papers (streaming from OpenAlex; downloading PDFs as needed)...")
            pbar_process = tqdm(total=MAX_PAPERS_PER_DOMAIN, initial=existing_count, desc=f"{domain}", unit="paper")

            stream = fetch_openalex(concept_id)
            exhausted = False

            while domain_processed < MAX_PAPERS_PER_DOMAIN and not exhausted:
                batch = []
                # Collect a batch of candidates
                while len(batch) < PROCESS_BATCH_SIZE:
                    try:
                        work = next(stream)
                    except StopIteration:
                        exhausted = True
                        break

                    concepts = work.get("concepts", [])
                    if not is_domain_first_concept(concepts, domain):
                        continue

                    pdf_candidates = _candidate_pdf_urls_from_work(work)
                    pdf_url = pdf_candidates[0] if pdf_candidates else None
                    if not pdf_url:
                        continue

                    openalex_id = extract_openalex_id(work)
                    if not openalex_id:
                        import hashlib
                        title = work.get("display_name", "")
                        year = work.get("publication_year", "")
                        fallback_id = hashlib.md5(f"{title}_{year}".encode()).hexdigest()[:12]
                        openalex_id = f"fallback_{fallback_id}"

                    if openalex_id in processed_ids:
                        continue

                    # Quick pre-filter: if OpenAlex provides an abstract and it's too short, skip.
                    abstract = work.get("abstract")
                    if not abstract:
                        inverted_index = work.get("abstract_inverted_index")
                        if inverted_index:
                            abstract = reconstruct_abstract_from_inverted_index(inverted_index)
                    if abstract and abstract.strip() and abstract_word_count(abstract) < MIN_ABSTRACT_WORDS:
                        processed_ids.add(openalex_id)
                        continue

                    processed_ids.add(openalex_id)
                    batch.append((work, domain, openalex_id, pdf_url))

                if not batch:
                    break

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(process_paper_download, paper_data) for paper_data in batch]
                    for future in as_completed(futures):
                        if domain_processed >= MAX_PAPERS_PER_DOMAIN:
                            break
                        try:
                            metadata = future.result()
                        except Exception:
                            metadata = None

                        if not metadata:
                            continue

                        # Only count papers that meet abstract-length requirement (enforced in process_paper_download).
                        existing = existing_papers.get(metadata["id"])
                        if (existing is None) or (not _is_valid_existing(existing)):
                            meta_out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                            meta_out.flush()
                            existing_papers[metadata["id"]] = metadata
                            domain_processed += 1
                            total_processed += 1
                            pbar_process.update(1)
                            if domain_processed % 5 == 0 or domain_processed == MAX_PAPERS_PER_DOMAIN:
                                print(f"  {domain}: {domain_processed}/{MAX_PAPERS_PER_DOMAIN} valid abstracts")

            pbar_process.close()

            if domain_processed < MAX_PAPERS_PER_DOMAIN:
                print(
                    f"  Warning: Could not reach {MAX_PAPERS_PER_DOMAIN} valid papers for {domain}. "
                    f"Got {domain_processed}/{MAX_PAPERS_PER_DOMAIN}."
                )
            else:
                print(
                    f"  ✅ {domain}: Reached target of {MAX_PAPERS_PER_DOMAIN} papers "
                    f"({existing_count} from earlier run, {domain_processed - existing_count} new)"
                )
    
    print(f"\n✅ Done! Total processed: {total_processed} papers across all domains.")
    return True


# ==========================
# BACKFILL: FULL TEXT FROM EXISTING METADATA
# ==========================
def _ensure_pdf_and_maybe_fulltext_for_metadata_record(paper: dict) -> tuple[bool, bool]:
    """
    Given a metadata.jsonl record, ensure pdf + extracted full text exist on disk.
    Returns (pdf_ok, fulltext_ok).
    """
    pid = paper.get("id")
    domain = paper.get("domain")
    pdf_url = paper.get("pdf_url")
    openalex_id = paper.get("openalex_id")
    abstract = paper.get("abstract") or ""

    if not pid or not domain:
        return (False, False)
    if abstract_word_count(abstract) < MIN_ABSTRACT_WORDS:
        return (False, False)

    domain_dir = os.path.join(OUTPUT_DIR, domain)
    pdf_dir = os.path.join(domain_dir, "pdfs")
    text_dir = os.path.join(domain_dir, "text")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
    txt_path = os.path.join(text_dir, f"{pid}.txt")

    pdf_ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
    fulltext_ok = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0
    if pdf_ok and (fulltext_ok or not EXTRACT_FULLTEXT):
        return (pdf_ok, fulltext_ok)

    # Ensure PDF exists
    if not pdf_ok:
        # Try stored pdf_url first, then fall back to OpenAlex best locations if available.
        urls = []
        if pdf_url:
            urls.append(pdf_url)
        if openalex_id:
            work = fetch_openalex_work(openalex_id)
            if work:
                urls.extend(_candidate_pdf_urls_from_work(work))

        tried = set()
        downloaded = False
        for u in urls:
            if not u or u in tried:
                continue
            tried.add(u)
            if download_pdf(u, pdf_path):
                downloaded = True
                break

        if not downloaded:
            return (False, False)
        pdf_ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
        if not pdf_ok:
            return (False, False)

    # Extract full text
    if not EXTRACT_FULLTEXT:
        return (pdf_ok, fulltext_ok)

    # Best-effort full-text extraction
    tei_xml = grobid_extract(pdf_path)
    if tei_xml and tei_xml.strip():
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tei_xml)
        except Exception:
            pass
    else:
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = "\n".join(text_parts)
            if text.strip():
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
        except Exception:
            pass

    fulltext_ok = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0
    return (pdf_ok, fulltext_ok)


def backfill_pdfs_from_metadata(max_workers: int = MAX_WORKERS) -> bool:
    """
    For every metadata.jsonl record with abstract >= MIN_ABSTRACT_WORDS, ensure we have:
    - papers/{domain}/pdfs/{id}.pdf
    Optionally (EXTRACT_FULLTEXT=True):
    - papers/{domain}/text/{id}.txt  (full text; best-effort)
    """
    if not os.path.exists(META_FILE):
        print(f"Error: {META_FILE} not found")
        return False

    records = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except Exception:
                continue
            if abstract_word_count(paper.get("abstract") or "") < MIN_ABSTRACT_WORDS:
                continue
            records.append(paper)

    if not records:
        print("No eligible records found for backfill.")
        return True

    # Only backfill records that are missing PDFs
    missing_records = []
    for paper in records:
        pid = paper.get("id")
        domain = paper.get("domain")
        if not pid or not domain:
            continue
        pdf_path = os.path.join(OUTPUT_DIR, domain, "pdfs", f"{pid}.pdf")
        if not (os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0):
            missing_records.append(paper)

    print(
        f"Backfilling PDFs for {len(missing_records)}/{len(records)} papers "
        f"(abstracts >= {MIN_ABSTRACT_WORDS} words)..."
    )
    if len(missing_records) == 0:
        print("All eligible papers already have PDFs.")
        return True

    pbar = tqdm(total=len(missing_records), desc="pdf backfill", unit="paper")
    pdf_ok_count = 0
    fulltext_ok_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_ensure_pdf_and_maybe_fulltext_for_metadata_record, paper)
            for paper in missing_records
        ]
        for future in as_completed(futures):
            try:
                pdf_ok, fulltext_ok = future.result()
            except Exception:
                pdf_ok, fulltext_ok = (False, False)
            if pdf_ok:
                pdf_ok_count += 1
            if fulltext_ok:
                fulltext_ok_count += 1
            pbar.update(1)
    pbar.close()
    print(
        f"Backfill complete. PDFs downloaded for {pdf_ok_count}/{len(missing_records)} papers "
        f"(some may have failed due to PDF access)."
    )
    return True


# ==========================
# BACKFILL: PDFs FOR ALL EXISTING ABSTRACT FILES
# ==========================
def backfill_pdfs_for_abstract_files(max_workers: int = MAX_WORKERS, max_to_attempt: Optional[int] = None) -> bool:
    """
    Ensure that for every existing abstract file on disk:
      papers/{collection}/{domain}/abstracts/{id}.txt
    we have a corresponding PDF:
      papers/{collection}/{domain}/pdfs/{id}.pdf

    Uses metadata_{collection}.jsonl to find `pdf_url` and `openalex_id`, and falls back to
    OpenAlex best locations when direct PDF download fails (paywalls may still block).
    """
    # Index metadata by (domain, id) using the last occurrence
    latest: dict = {}
    if not os.path.exists(META_FILE):
        print(f"Error: {META_FILE} not found")
        return False

    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            dom = rec.get("domain")
            pid = rec.get("id")
            if dom and pid:
                latest[(dom, pid)] = rec

    tasks: list[tuple[str, str]] = []
    for domain in CONCEPTS.keys():
        abstract_dir = os.path.join(OUTPUT_DIR, domain, "abstracts")
        pdf_dir = os.path.join(OUTPUT_DIR, domain, "pdfs")
        if not os.path.isdir(abstract_dir):
            continue
        os.makedirs(pdf_dir, exist_ok=True)
        for p in Path(abstract_dir).glob("*.txt"):
            pid = p.stem
            pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                continue
            tasks.append((domain, pid))

    if max_to_attempt is not None:
        tasks = tasks[: max(0, int(max_to_attempt))]

    print(f"Backfilling PDFs for {len(tasks)} abstract files missing PDFs...")
    if not tasks:
        print("All abstract files already have PDFs.")
        return True

    failures_path = os.path.join(OUTPUT_DIR, "pdf_backfill_failures.jsonl")
    lock = Lock()
    downloaded = 0
    failed = 0

    def _download_one(domain: str, pid: str) -> bool:
        rec = latest.get((domain, pid))
        if not rec:
            return False
        pdf_dir = os.path.join(OUTPUT_DIR, domain, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")

        urls = []
        if rec.get("pdf_url"):
            urls.append(rec["pdf_url"])
        openalex_id = rec.get("openalex_id")
        if openalex_id:
            work = fetch_openalex_work(openalex_id)
            if work:
                urls.extend(_candidate_pdf_urls_from_work(work))

        # unique
        seen = set()
        for u in urls:
            if not u or u in seen:
                continue
            seen.add(u)
            if download_pdf(u, pdf_path):
                return True
        return False

    pbar = tqdm(total=len(tasks), desc="pdf backfill (from abstracts)", unit="paper")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, d, pid): (d, pid) for d, pid in tasks}
        for future in as_completed(futures):
            domain, pid = futures[future]
            try:
                ok = future.result()
            except Exception:
                ok = False

            with lock:
                if ok:
                    downloaded += 1
                else:
                    failed += 1
                    try:
                        rec = latest.get((domain, pid), {})
                        with open(failures_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "domain": domain,
                                        "id": pid,
                                        "openalex_id": rec.get("openalex_id"),
                                        "pdf_url": rec.get("pdf_url"),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                pbar.update(1)
    pbar.close()

    print(f"PDF backfill complete: downloaded={downloaded}, failed={failed}")
    if failed:
        print(f"Failures written to: {failures_path}")
    return True


# ==========================
# ENRICH: ADD PUBLICATION DATE/MONTH INTO METADATA JSONL
# ==========================
def enrich_metadata_dates() -> bool:
    """
    Rewrite the current collection's metadata file so each record includes:
    - publication_date (YYYY-MM-DD) when available from OpenAlex
    - publication_month (YYYY-MM) derived from publication_date

    This preserves the JSONL format and line order (no dedup).
    """
    if not os.path.exists(META_FILE):
        print(f"Error: {META_FILE} not found")
        return False

    cache_path = os.path.join(OUTPUT_DIR, "openalex_publication_dates_cache.json")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                date_cache = json.load(f) or {}
        else:
            date_cache = {}
    except Exception:
        date_cache = {}

    def _cache_get(openalex_id: str):
        return date_cache.get(openalex_id)

    def _cache_set(openalex_id: str, pub_date: str):
        date_cache[openalex_id] = pub_date

    tmp_path = META_FILE + ".tmp"
    bak_path = META_FILE + ".bak"

    updated = 0
    fetched = 0
    total = 0

    with open(META_FILE, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            total += 1
            try:
                rec = json.loads(raw)
            except Exception:
                # Keep unparseable lines as-is
                fout.write(line)
                continue

            openalex_id = rec.get("openalex_id")
            pub_date = rec.get("publication_date")

            if not pub_date and openalex_id:
                cached = _cache_get(openalex_id)
                if cached:
                    pub_date = cached
                else:
                    work = fetch_openalex_work(openalex_id)
                    d = extract_publication_date(work) if work else None
                    if d:
                        pub_date = d
                        _cache_set(openalex_id, d)
                        fetched += 1

            pub_month = publication_month_from_date(pub_date) if pub_date else None

            # Only count as updated if we added something new
            if pub_date and rec.get("publication_date") != pub_date:
                rec["publication_date"] = pub_date
                updated += 1
            if pub_month and rec.get("publication_month") != pub_month:
                rec["publication_month"] = pub_month
                updated += 1

            # If year is missing but publication_date exists, fill it.
            if (rec.get("year") is None) and pub_date:
                try:
                    rec["year"] = int(pub_date.split("-")[0])
                except Exception:
                    pass

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(date_cache, f, indent=2, sort_keys=True)
    except Exception:
        pass

    # Backup and replace atomically
    try:
        if not os.path.exists(bak_path):
            shutil.copy2(META_FILE, bak_path)
    except Exception:
        pass
    os.replace(tmp_path, META_FILE)

    print(f"Enriched metadata dates: updated_fields={updated}, fetched_from_openalex={fetched}, total_lines={total}")
    print(f"Metadata: {META_FILE}")
    print(f"Date cache: {cache_path}")
    if os.path.exists(bak_path):
        print(f"Backup: {bak_path}")
    return True


# Note: All papers in metadata.jsonl are already filtered for #1 concept during scraping
# No need for additional filtering step


def main():
    """Main scraping pipeline."""
    parser = argparse.ArgumentParser(
        description="Scrape papers from OpenAlex where domain is #1 concept (default behavior)."
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Output subfolder under papers/ (e.g., 2015_back, 2025_back_2023).",
    )
    parser.add_argument(
        "--from-date",
        dest="from_date",
        default=None,
        help="Inclusive start publication date (YYYY-MM-DD), e.g. 2023-01-01",
    )
    parser.add_argument(
        "--to-date",
        dest="to_date",
        default=None,
        help="Inclusive end publication date (YYYY-MM-DD), e.g. 2025-12-31",
    )
    parser.add_argument(
        "--require-pdf",
        action="store_true",
        help="Only record/count papers when PDF download succeeds (recommended for date-range collections).",
    )
    parser.add_argument(
        "--no-require-pdf",
        action="store_true",
        help="Allow recording abstracts even when PDF download fails (default for non-range scraping).",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape papers from OpenAlex (same as default, kept for compatibility)"
    )
    parser.add_argument(
        "--backfill-pdfs",
        action="store_true",
        help=f"Download PDFs for all metadata records with abstracts >= {MIN_ABSTRACT_WORDS} words",
    )
    parser.add_argument(
        "--backfill-pdfs-for-abstracts",
        action="store_true",
        help="Download PDFs for every existing abstract file that is missing a PDF (best-effort; paywalls may block).",
    )
    parser.add_argument(
        "--max-pdf-attempts",
        type=int,
        default=None,
        help="Limit the number of missing-PDF abstract files to attempt (debug).",
    )
    parser.add_argument(
        "--enrich-metadata-dates",
        action="store_true",
        help="Rewrite metadata jsonl to include publication_date and publication_month (cached OpenAlex lookups).",
    )
    args = parser.parse_args()
    configure_output(args.collection)
    configure_date_range(args.from_date, args.to_date)
    # Default behavior: if a date range is specified, require PDFs unless explicitly disabled.
    if args.require_pdf:
        configure_require_pdf(True)
    elif args.no_require_pdf:
        configure_require_pdf(False)
    else:
        configure_require_pdf(bool(args.from_date or args.to_date))
    
    if args.backfill_pdfs:
        backfill_pdfs_from_metadata()
        return
    if args.backfill_pdfs_for_abstracts:
        backfill_pdfs_for_abstract_files(max_to_attempt=args.max_pdf_attempts)
        return
    if args.enrich_metadata_dates:
        enrich_metadata_dates()
        return

    # Default behavior: scrape papers where domain is #1 (and ensure full text exists)
    scrape_papers()


if __name__ == "__main__":
    main()
