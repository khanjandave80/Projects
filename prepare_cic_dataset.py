import csv
import hashlib
import os
import shutil
import zipfile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "CIC-Trap4Phish-2025")
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset", "cic_attachment_manifest.csv")

ARCHIVES = {
    "Malicious_Word.zip": ("malicious", "word"),
    "Benign_Word.zip": ("benign", "word"),
    "PDF_Benign.zip": ("benign", "pdf"),
    "Malicious_PDF.zip": ("malicious", "pdf"),
    "Malicious_HTML.zip": ("malicious", "html"),
    "Benign_HTML.zip": ("benign", "html"),
    "Malicious_Excel.zip": ("malicious", "excel"),
    "Benign_Excel.zip": ("benign", "excel"),
}

FIELDNAMES = [
    "id",
    "original_filename",
    "file_path",
    "relative_path",
    "file_extension",
    "file_format",
    "attachment_type",
    "attachment_label",
    "file_size_bytes",
    "sha256",
    "extraction_status",
    "source_dataset",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive_path, label, file_format):
    destination = os.path.join(OUTPUT_DIR, label, file_format)
    os.makedirs(destination, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            original_filename = os.path.basename(member.filename)
            if not original_filename:
                continue
            target = os.path.join(destination, original_filename)
            if os.path.exists(target):
                stem, extension = os.path.splitext(original_filename)
                target = os.path.join(destination, f"{stem}_{len(extracted) + 1}{extension}")
            try:
                if not os.path.isfile(target):
                    with archive.open(member) as source, open(target, "wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            except OSError:
                pass
            extracted.append((target, os.path.basename(target)))
    return extracted


def main():
    rows = []
    row_id = 1
    for archive_name, (label, file_format) in ARCHIVES.items():
        archive_path = os.path.join(DOWNLOADS_DIR, archive_name)
        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f"Missing CIC archive: {archive_path}")
        files = safe_extract(archive_path, label, file_format)
        for file_path, filename in files:
            relative_path = os.path.relpath(file_path, BASE_DIR).replace(os.sep, "/")
            try:
                file_size = os.path.getsize(file_path)
                extraction_status = "extracted"
            except OSError:
                file_size = ""
                extraction_status = "blocked_or_quarantined"
            rows.append(
                {
                    "id": row_id,
                    "original_filename": filename,
                    "file_path": file_path,
                    "relative_path": relative_path,
                    "file_extension": os.path.splitext(filename)[1].lower(),
                    "file_format": file_format,
                    "attachment_type": label,
                    "attachment_label": int(label == "malicious"),
                    "file_size_bytes": file_size,
                    "sha256": "",
                    "extraction_status": extraction_status,
                    "source_dataset": "CIC-Trap4Phish-2025",
                }
            )
            row_id += 1

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extracted {len(rows):,} files")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()