import os
import random
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DIKE_LABELS = os.path.join(DATASET_DIR, "DikeDataset-main", "labels")

# Realistic Name Templates
MALWARE_NAMES = [
    ("Invoice_INV-{num}.pdf.exe", "Invoice / Billing lure with hidden executable extension"),
    ("DHL_Shipping_Receipt_{num}.zip", "Compressed archive containing obfuscated loader"),
    ("Resume_Candidate_{num}.docm", "Word document with weaponized VBA auto-run macros"),
    ("Windows_KB503_{num}_Update.exe", "Fake system security patch / trojan dropper"),
    ("Bank_Remittance_Advice_{num}.vbs", "VBScript disguised as banking notification"),
    ("Salary_Hike_Revision_2024_{num}.scr", "Screensaver binary executable disguised as corporate policy"),
    ("Payment_Confirmation_{num}.iso", "Disk image bypass container hiding ransomware payload"),
    ("Tax_Assessment_Refund_{num}.xlsm", "Excel spreadsheet embedding malicious macro shellcode")
]

BENIGN_NAMES = [
    ("Quarterly_Report_Q{quarter}.pdf", "Standard PDF financial report"),
    ("Meeting_Minutes_{num}.docx", "Standard Microsoft Word clean document"),
    ("Project_Roadmap_2025.xlsx", "Standard clean Excel spreadsheet"),
    ("Company_Security_Policy_v{num}.pdf", "Corporate compliance guideline PDF"),
    ("Zoom_Client_Installer_{num}.exe", "Digitally signed verified application installer"),
    ("Team_Photo_Archive_{num}.zip", "Compressed photo and media archive"),
    ("Vendor_Contract_Agreement_{num}.pdf", "Counter-signed PDF business contract"),
    ("System_Diagnostics_Tool.exe", "Verified Microsoft-signed management utility")
]

# Load Dike labels
benign_df = pd.read_csv(os.path.join(DIKE_LABELS, "benign.csv"))
malware_df = pd.read_csv(os.path.join(DIKE_LABELS, "malware.csv"))

# Balance: Take all 1,082 benign and 1,500 malware
benign_df = benign_df.sample(n=len(benign_df), random_state=42)
malware_df = malware_df.sample(n=min(1500, len(malware_df)), random_state=42)

records = []

# Process Benign
for idx, row in benign_df.iterrows():
    template, desc = random.choice(BENIGN_NAMES)
    filename = template.format(num=random.randint(1000, 9999), quarter=random.randint(1, 4))
    ext = os.path.splitext(filename)[1]
    
    card_text = (
        f"Attachment: {filename}. Extension: {ext}. "
        f"File Type: Benign User Document/Executable. "
        f"Digital Signature: Valid & Verified Publisher. "
        f"Entropy: {random.uniform(3.2, 5.8):.2f} (Normal). "
        f"Behavioral Scan: No malicious API hooks or obfuscated macros detected. "
        f"Status: Safe legitimate attachment."
    )
    records.append({
        "filename": filename,
        "extension": ext,
        "hash": row["hash"],
        "text": card_text,
        "label": 0
    })

# Process Malware
for idx, row in malware_df.iterrows():
    template, desc = random.choice(MALWARE_NAMES)
    filename = template.format(num=random.randint(1000, 9999))
    ext = os.path.splitext(filename)[1]
    
    malice = float(row.get("malice", 0.85))
    is_double_ext = ".pdf.exe" in filename or filename.endswith(".vbs") or filename.endswith(".scr")
    
    card_text = (
        f"Attachment: {filename}. Extension: {ext}. "
        f"File Type: Potentially Harmful Payload ({desc}). "
        f"Digital Signature: Unsigned or Invalid Certificate. "
        f"Entropy: {random.uniform(6.8, 7.95):.2f} (High - Obfuscated/Packed). "
        f"Behavioral Scan: Malice rating={malice:.2f}, double_extension={is_double_ext}, "
        f"Suspicious calls detected: Memory injection, autorun persistence, network beaconing."
    )
    records.append({
        "filename": filename,
        "extension": ext,
        "hash": row["hash"],
        "text": card_text,
        "label": 1
    })

df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Train/Test Split (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

out_all = os.path.join(DATASET_DIR, "m5_malware_dataset.csv")
out_train = os.path.join(DATASET_DIR, "m5_malware_train.csv")
out_test = os.path.join(DATASET_DIR, "m5_malware_test.csv")

df.to_csv(out_all, index=False)
train_df.to_csv(out_train, index=False)
test_df.to_csv(out_test, index=False)

print(f"Successfully generated {len(df)} total samples!")
print(f"Train split saved to: {out_train} ({len(train_df)} samples)")
print(f"Test split saved to:  {out_test}  ({len(test_df)} samples)")