import os
import hashlib
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Silence pypdf verbose logs
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset", "cic_attachment_manifest.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "cic_attachment_text.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Set to False to skip SHA256 (much faster, not needed for training)
COMPUTE_SHA256 = False

# Number of parallel workers (use CPU cores - 1)
NUM_WORKERS = max(1, cpu_count() - 1)


def sha256_file(path):
    if not COMPUTE_SHA256:
        return ""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return ""


def extract_pdf(path):
    if PdfReader is None:
        return "", "unsupported_format"
    try:
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts), "extracted"
    except Exception as e:
        return "", f"extraction_failed: {e}"


def extract_docx(path):
    if Document is None:
        return "", "unsupported_format"
    try:
        doc = Document(path)
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(text_parts), "extracted"
    except Exception as e:
        return "", f"extraction_failed: {e}"


def extract_xlsx(path):
    if load_workbook is None:
        return "", "unsupported_format"
    try:
        wb = load_workbook(path, read_only=True, data_only=True)
        text_parts = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                row_text = " ".join(str(v) for v in row if v is not None)
                if row_text.strip():
                    text_parts.append(row_text)
        wb.close()
        return "\n".join(text_parts), "extracted"
    except Exception as e:
        return "", f"extraction_failed: {e}"


def extract_html(path):
    if BeautifulSoup is None:
        return "", "unsupported_format"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text, "extracted"
    except Exception as e:
        return "", f"extraction_failed: {e}"


def extract_text_by_format(path, file_format):
    if not os.path.isfile(path):
        return "", "missing_file"
    file_format = (file_format or "").lower()
    if file_format == "pdf":
        return extract_pdf(path)
    elif file_format == "word":
        return extract_docx(path)
    elif file_format == "excel":
        return extract_xlsx(path)
    elif file_format == "html":
        return extract_html(path)
    else:
        return "", "unsupported_format"


def process_row(row):
    """Process a single row - designed for multiprocessing."""
    file_id = str(row["id"])
    file_path = row["file_path"]
    file_format = row["file_format"]
    attachment_label = row["attachment_label"]

    if not os.path.isfile(file_path):
        extracted_text = ""
        extraction_status = "missing_file"
        sha256 = ""
    else:
        extracted_text, extraction_status = extract_text_by_format(file_path, file_format)
        if extraction_status == "extracted" and not extracted_text.strip():
            extraction_status = "empty_text"
        sha256 = sha256_file(file_path)

    return {
        "id": row["id"],
        "original_filename": row["original_filename"],
        "file_path": row["file_path"],
        "relative_path": row["relative_path"],
        "file_format": row["file_format"],
        "attachment_label": int(attachment_label) if pd.notna(attachment_label) else -1,
        "extracted_text": extracted_text,
        "extraction_status": extraction_status,
        "source_dataset": row["source_dataset"],
        "sha256": sha256
    }


def load_existing_output():
    if not os.path.exists(OUTPUT_CSV):
        return set(), []
    try:
        existing = pd.read_csv(OUTPUT_CSV)
        processed_ids = set(existing["id"].astype(str).tolist())
        existing_rows = existing.to_dict("records")
        print(f"Resuming: {len(processed_ids):,} already extracted")
        return processed_ids, existing_rows
    except Exception:
        return set(), []


def main():
    print(f"Reading manifest: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    total_files = len(df)
    print(f"Total files in manifest: {total_files:,}")
    print(f"Using {NUM_WORKERS} parallel workers")
    print(f"SHA256 computation: {'ON' if COMPUTE_SHA256 else 'OFF (faster)'}")

    processed_ids, existing_rows = load_existing_output()

    # Filter out already processed
    todo_rows = [row for _, row in df.iterrows() if str(row["id"]) not in processed_ids]
    print(f"Remaining to process: {len(todo_rows):,}")

    if not todo_rows:
        print("All files already processed!")
        return

    all_rows = existing_rows.copy()

    # Process in parallel with progress bar
    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_row, todo_rows, chunksize=50),
                          total=len(todo_rows), desc="Extracting", unit="file"):
            all_rows.append(result)

            # Periodic save
            if len(all_rows) % 2000 == 0:
                pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)

    # Final save
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total = len(out_df)
    extracted = (out_df["extraction_status"] == "extracted").sum()
    missing = (out_df["extraction_status"] == "missing_file").sum()
    unsupported = (out_df["extraction_status"] == "unsupported_format").sum()
    failed = out_df["extraction_status"].str.startswith("extraction_failed").sum()
    empty_text = (out_df["extraction_status"] == "empty_text").sum()
    benign = (out_df["attachment_label"] == 0).sum()
    malicious = (out_df["attachment_label"] == 1).sum()

    word = (out_df["file_format"] == "word").sum()
    excel = (out_df["file_format"] == "excel").sum()
    pdf = (out_df["file_format"] == "pdf").sum()
    html = (out_df["file_format"] == "html").sum()

    print(f"\nTotal rows: {total:,}")
    print(f"Extracted: {extracted:,}")
    print(f"Missing: {missing:,}")
    print(f"Unsupported: {unsupported:,}")
    print(f"Failed: {failed:,}")
    print(f"Empty text: {empty_text:,}")
    print(f"Benign: {benign:,}")
    print(f"Malicious: {malicious:,}")
    print(f"Word: {word:,}")
    print(f"Excel: {excel:,}")
    print(f"PDF: {pdf:,}")
    print(f"HTML: {html:,}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()