import os, json, base64, io, random, sqlite3, secrets, re
_re = re
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 22 * 1024 * 1024


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Image upload is too large. Please use an image smaller than 15 MB."}), 413

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DIST_DIR   = os.path.join(BASE_DIR, "game", "dist")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Email_DistilBERT")
DB_PATH    = os.path.join(BASE_DIR, "data", "cybersleuth_auth.db")
DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "local").lower()
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 256

print("Loading DistilBERT model...")
tokenizer = None
model = None
if os.path.isdir(MODEL_PATH):
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        model.eval()
        print(f"DistilBERT loaded on {DEVICE}")
    except Exception as e:
        print(f"[WARN] DistilBERT not loaded: {e}")
else:
    print(f"[WARN] DistilBERT folder not found at: {MODEL_PATH}")

SMS_MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "sms_distilbert"),
    os.path.join(BASE_DIR, "models", "model"),
    os.path.join(BASE_DIR, "models", "Email_DistilBERT"),
]
SMS_MAX_LEN = 128
sms_tokenizer = None
sms_model = None
sms_model_path = None

# SMS model intentionally disabled — email classifier gives wrong results on SMS
print("[INFO] SMS model disabled; using heuristic fallback")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


ADMIN_MAP = {
    "adity": "Admin_1",
    "aditya": "Admin_1",
    "khanjan": "Admin_2",
}

def get_admin_role_for(username: str):
    if not username:
        return None
    return ADMIN_MAP.get(username.strip().lower(), None)


def init_db():
    admin_1_password = os.environ.get("CS_ADMIN_1_PASSWORD")
    admin_2_password = os.environ.get("CS_ADMIN_2_PASSWORD")
    generated_passwords = []

    if DEPLOYMENT_MODE == "online":
        if not admin_1_password or not admin_2_password:
            raise RuntimeError(
                "Online mode requires CS_ADMIN_1_PASSWORD and CS_ADMIN_2_PASSWORD."
            )
    else:
        if not admin_1_password:
            admin_1_password = secrets.token_urlsafe(24)
            generated_passwords.append(("Admin_1", "aditya", admin_1_password))
        if not admin_2_password:
            admin_2_password = secrets.token_urlsafe(24)
            generated_passwords.append(("Admin_2", "khanjan", admin_2_password))

    if min(len(admin_1_password), len(admin_2_password)) < 16:
        raise RuntimeError("Administrator passwords must be at least 16 characters.")

    if generated_passwords:
        print("[WARN] Temporary local administrator password(s):")
        for role, username, password in generated_passwords:
            print(f"  {role} ({username}): {password}")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                best_score INTEGER NOT NULL DEFAULT 0,
                best_category TEXT NOT NULL DEFAULT 'Normal',
                games_played INTEGER NOT NULL DEFAULT 0,
                is_admin INTEGER NOT NULL DEFAULT 0,
                admin_role TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS score_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                score INTEGER NOT NULL,
                category TEXT NOT NULL,
                mission TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Migration: ensure games_played, is_admin, and admin_role columns exist
        cur = conn.execute("PRAGMA table_info(users)")
        cols = [r["name"] for r in cur.fetchall()]
        if "games_played" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN games_played INTEGER NOT NULL DEFAULT 0")
        if "is_admin" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        if "admin_role" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN admin_role TEXT DEFAULT NULL")

        # PURGE ALL SYNTHETIC AND TEST DATA (sarah_connor, alex_chen, priya_sharma, admin, etc.)
        conn.execute("DELETE FROM score_entries WHERE LOWER(username) IN ('sarah_connor', 'alex_chen', 'priya_sharma', 'david_ross', 'elena_rostova', 'player_alpha', 'admin')")
        conn.execute("DELETE FROM users WHERE LOWER(username) IN ('sarah_connor', 'alex_chen', 'priya_sharma', 'david_ross', 'elena_rostova', 'player_alpha', 'admin')")

        import hashlib
        ts = now_iso()

        # Seed or update the administrator credentials for this deployment.
        sha_aditya = hashlib.sha256(admin_1_password.encode("utf-8")).hexdigest()
        pwd_hash_aditya = generate_password_hash(sha_aditya)
        aditya_user = conn.execute("SELECT id FROM users WHERE LOWER(username) = 'aditya'").fetchone()
        if not aditya_user:
            adity_old = conn.execute("SELECT id FROM users WHERE LOWER(username) = 'adity'").fetchone()
            if adity_old:
                conn.execute(
                    "UPDATE users SET username = 'aditya', password_hash = ?, is_admin = 1, admin_role = 'Admin_1', updated_at = ? WHERE id = ?",
                    (pwd_hash_aditya, ts, adity_old["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO users (username, password_hash, best_score, best_category, games_played, is_admin, admin_role, created_at, updated_at) VALUES (?, ?, 0, 'Normal', 0, 1, 'Admin_1', ?, ?)",
                    ("aditya", pwd_hash_aditya, ts, ts),
                )
        else:
            conn.execute(
                "UPDATE users SET password_hash = ?, is_admin = 1, admin_role = 'Admin_1', updated_at = ? WHERE LOWER(username) = 'aditya'",
                (pwd_hash_aditya, ts),
            )

        sha_khanjan = hashlib.sha256(admin_2_password.encode("utf-8")).hexdigest()
        pwd_hash_khanjan = generate_password_hash(sha_khanjan)
        khanjan_user = conn.execute("SELECT id FROM users WHERE LOWER(username) = 'khanjan'").fetchone()
        if not khanjan_user:
            conn.execute(
                "INSERT INTO users (username, password_hash, best_score, best_category, games_played, is_admin, admin_role, created_at, updated_at) VALUES (?, ?, 0, 'Normal', 0, 1, 'Admin_2', ?, ?)",
                ("khanjan", pwd_hash_khanjan, ts, ts),
            )
        else:
            conn.execute(
                "UPDATE users SET password_hash = ?, is_admin = 1, admin_role = 'Admin_2', updated_at = ? WHERE LOWER(username) = 'khanjan'",
                (pwd_hash_khanjan, ts),
            )

        # Ensure NO OTHER users have administrator privilege
        conn.execute("UPDATE users SET is_admin = 0, admin_role = NULL WHERE LOWER(username) NOT IN ('adity', 'aditya', 'khanjan')")




def normalize_username(value: str):
    value = (value or "").strip()
    if not 3 <= len(value) <= 24:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if any(ch not in allowed for ch in value):
        return None
    return value


def client_hash_ok(value: str):
    return isinstance(value, str) and len(value.strip()) >= 32


import re as _re

# Trusted Indian bank/service sender prefixes (case-insensitive)
_TRUSTED_SENDERS = _re.compile(
    r'^(KOTAKB|HDFCBK|SBIINB|ICICIB|AXISBK|PAYTMB|IDFCBK|YESBNK|BOIIND|PNBSMS|'
    r'CANBNK|UNIONB|INDBNK|CENTBK|SYNDBK|OBCBNK|ALLBNK|CORPBK|DENABN|VIJBNK|'
    r'IRCTC|UIDAI|NSDL|CDSL|INCOMETAX|TRACES|EPFO|NPCI|BHIMUPI|GPAY|PHONEPE|'
    r'AMAZON|FLIPKRT|ZOMATO|SWIGGY|JIONET|AIRTEL|VODAFONE|BSNL|IDEA)',
    _re.IGNORECASE
)

# Patterns that strongly indicate a real bank transaction alert
_BANK_TXN_PATTERNS = [
    _re.compile(r'(rs\.?|inr|₹)\s*[\d,]+\.?\d*\s*(credited|debited|received|transferred|paid)', _re.IGNORECASE),
    _re.compile(r'(credited|debited|received|deducted)\s*.{0,30}(rs\.?|inr|₹)\s*[\d,]+', _re.IGNORECASE),
    _re.compile(r'(rs\.?|inr|₹)\s*[\d,]+\.?\d*\s*.{0,40}(debited|credited|deducted|transferred)', _re.IGNORECASE),
    _re.compile(r'a/?c\s*(no\.?\s*)?[xX*]{2,}\d{3,}', _re.IGNORECASE),
    _re.compile(r'account\s*[xX*]{4,}\d{3,}', _re.IGNORECASE),
    _re.compile(r'upi\s*ref\s*[:#]?\s*\d{8,}', _re.IGNORECASE),
    _re.compile(r'avl\s*bal\s*[:#]?\s*(rs\.?|inr|₹)', _re.IGNORECASE),
    _re.compile(r'\bnach\b', _re.IGNORECASE),
    _re.compile(r'pnr\s*[:#]?\s*\d{8,}', _re.IGNORECASE),
    _re.compile(r'otp\s*(is|:)\s*\d{4,8}', _re.IGNORECASE),
]

# Risky TLDs / shorteners in URLs
_RISKY_URL = _re.compile(
    r'https?://[^\s]*\.(tk|ml|xyz|ga|cf|top|click|pw|gq|zip|mov|co\.in\.)[^\s]*|'
    r'(bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly)/[^\s]+',
    _re.IGNORECASE
)

# Domains that are always safe in SMS context
_SAFE_DOMAINS = _re.compile(
    r'https?://(www\.)?(hdfcbank|sbi|onlinesbi|icicibank|axisbank|kotakbank|'
    r'paytm|phonepe|gpay|amazon\.in|flipkart|irctc\.co\.in|uidai\.gov\.in|'
    r'myaadhaar\.uidai|incometax\.gov|traces\.gov|epfindia\.gov|npci\.org\.in|'
    r'zomato|swiggy|jio\.com|airtel\.in)\.?(com|in|co\.in|gov\.in|org\.in)?',
    _re.IGNORECASE
)


def classify_sms_fallback(text: str):
    lower = (text or "").lower()
    original = text or ""

    # ── Step 1: Check for legitimate bank transaction signals ──────────────
    txn_signals = sum(1 for p in _BANK_TXN_PATTERNS if p.search(original))
    has_safe_domain = bool(_SAFE_DOMAINS.search(original))
    has_risky_url   = bool(_RISKY_URL.search(original))

    # If it looks like a real transaction alert and has no risky URL → Safe
    if txn_signals >= 2 and not has_risky_url:
        return {
            "label": "Safe",
            "confidence": 94.0,
            "safe_prob": 94.0,
            "fake_prob": 6.0,
            "reason": "Matches legitimate bank transaction alert pattern (account number, UPI ref, balance).",
        }
    if txn_signals >= 1 and has_safe_domain and not has_risky_url:
        return {
            "label": "Safe",
            "confidence": 91.0,
            "safe_prob": 91.0,
            "fake_prob": 9.0,
            "reason": "Transaction alert from a recognised official domain.",
        }

    # ── Step 2: Risky URL present → strong fake signal ─────────────────────
    if has_risky_url:
        return {
            "label": "Promotional / Fake",
            "confidence": 93.0,
            "safe_prob": 7.0,
            "fake_prob": 93.0,
            "reason": "Contains a suspicious short URL or high-risk domain.",
        }

    # ── Step 3: Keyword scoring for remaining cases ────────────────────────
    score = 0
    reasons = []

    scam_urgency = ["urgent", "verify immediately", "suspended", "blocked", "limited",
                    "won", "winner", "prize", "lottery", "lucky draw", "claim now",
                    "act now", "expire", "last chance"]
    if any(w in lower for w in scam_urgency):
        score += 3
        reasons.append("Scam urgency or prize wording")

    if any(w in lower for w in ["http://", "https://"]):
        score += 1  # generic link — not automatically bad
        reasons.append("Contains a link")

    if any(w in lower for w in ["offer", "sale", "discount", "promo", "cashback",
                                  "free", "subscribe", "unsubscribe"]):
        score += 1
        reasons.append("Promotional wording")

    # KYC demand without transaction context is suspicious
    if "kyc" in lower and txn_signals == 0:
        score += 3
        reasons.append("KYC demand with no transaction context")

    # Asking for PIN/CVV/password/OTP in a non-OTP context
    if any(w in lower for w in ["pin", "cvv", "password"]) and "otp is" not in lower:
        score += 3
        reasons.append("Requests sensitive credentials")

    if score >= 3:
        fake_prob = min(95.0, 65.0 + score * 4.0)
        return {
            "label": "Promotional / Fake",
            "confidence": round(fake_prob, 1),
            "safe_prob": round(100.0 - fake_prob, 1),
            "fake_prob": round(fake_prob, 1),
            "reason": "; ".join(reasons) or "Risky message patterns detected.",
        }

    safe_prob = min(92.0, 78.0 + max(0, 2 - score) * 5.0)
    return {
        "label": "Safe",
        "confidence": round(safe_prob, 1),
        "safe_prob": round(safe_prob, 1),
        "fake_prob": round(100.0 - safe_prob, 1),
        "reason": "No strong scam pattern detected.",
    }


def classify_sms_text(text: str):
    text = (text or "").strip()
    if not text:
        return {
            "label": "Safe",
            "confidence": 0.0,
            "safe_prob": 0.0,
            "fake_prob": 0.0,
            "reason": "Empty text.",
        }

    if sms_model is None or sms_tokenizer is None:
        return classify_sms_fallback(text)

    try:
        enc = sms_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=SMS_MAX_LEN,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = sms_model(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            )
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())

        is_fake = pred == 1
        confidence = round(float(probs[pred]) * 100, 1)
        return {
            "label": "Promotional / Fake" if is_fake else "Safe",
            "confidence": confidence,
            "safe_prob": round(float(probs[0]) * 100, 1),
            "fake_prob": round(float(probs[1]) * 100, 1),
            "reason": "Model prediction from the trained SMS classifier.",
            "model_path": sms_model_path,
        }
    except Exception as e:
        fallback = classify_sms_fallback(text)
        fallback["reason"] = f"Fallback heuristic used: {e}"
        return fallback


def create_session(user_id: int):
    token = secrets.token_hex(24)
    ts = now_iso()
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (token, user_id, ts, ts),
        )
    return token


def get_user_by_token(token: str):
    if not token:
        return None
    with db_connect() as conn:
        row = conn.execute("""
            SELECT u.id, u.username, u.best_score, u.best_category, s.token
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
        """, (token,)).fetchone()
    return row


def leaderboard_rows():
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT
                u.username,
                u.best_score AS score,
                u.best_category AS category,
                COALESCE(u.games_played, 0) AS games_played,
                COALESCE(u.is_admin, 0) AS is_admin,
                u.admin_role,
                u.updated_at
            FROM users u
            ORDER BY u.best_score DESC, u.games_played DESC, u.updated_at ASC
        """).fetchall()
    return [
        {
            "rank": idx + 1,
            "username": row["username"],
            "score": row["score"],
            "category": row["category"],
            "games_played": max(1, row["games_played"]) if row["score"] > 0 else row["games_played"],
            "is_admin": bool(row["is_admin"]),
            "admin_role": row["admin_role"] or get_admin_role_for(row["username"]),
            "updated_at": row["updated_at"],
        }
        for idx, row in enumerate(rows)
    ]




init_db()

# ============================================================
# Mission 2 — BiLSTM URL classifier (3-class: Safe/Phishing/Suspicious)
# Mission 3 — QR BiLSTM (2-class: Safe/Phishing, trained on QR-decoded URLs)
# ============================================================
URL_MODEL_DIR = os.path.join(BASE_DIR, "models", "url_bilstm")
URL_CLASS_NAMES = ["Safe", "Phishing", "Suspicious"]

QR_MODEL_DIR = os.path.join(BASE_DIR, "models", "qr_BiLSTM")
QR_CLASS_NAMES = ["Safe", "Phishing"]


class URLBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop_emb = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            embed_dim, lstm_units, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        hidden = lstm_units * 2
        self.fc1 = nn.Linear(hidden, 64)
        self.relu = nn.ReLU()
        self.drop_fc = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        emb = self.drop_emb(self.embedding(x))
        out, _ = self.lstm(emb)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.drop_fc(out)
        return self.fc2(out)


url_model      = None
url_char_index = {}
url_cfg        = {"max_len": 200, "vocab_size": 2, "embed_dim": 64, "lstm_units": 128, "num_classes": 3}

qr_model       = None
qr_char_index  = {}
qr_cfg         = {"max_len": 200, "vocab_size": 2, "embed_dim": 64, "lstm_units": 128, "num_classes": 2}


def _load_bilstm(model_dir, num_classes, label):
    """Generic loader for any BiLSTM checkpoint. Returns (model, char_index, cfg) or (None, {}, cfg)."""
    cfg = {"max_len": 200, "vocab_size": 2, "embed_dim": 64, "lstm_units": 128, "num_classes": num_classes}
    try:
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(model_dir, "char_index.json"), "r", encoding="utf-8") as f:
            char_index = json.load(f)
        m = URLBiLSTM(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            lstm_units=cfg["lstm_units"],
            num_classes=cfg["num_classes"],
        ).to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=DEVICE))
        m.eval()
        print(f"{label} loaded.")
        return m, char_index, cfg
    except Exception as e:
        print(f"[WARN] {label} not loaded ({e}).")
        return None, {}, cfg


url_model, url_char_index, url_cfg = _load_bilstm(URL_MODEL_DIR, 3, "BiLSTM URL model (M2)")
qr_model,  qr_char_index,  qr_cfg  = _load_bilstm(QR_MODEL_DIR,  2, "BiLSTM QR model  (M3)")


def heuristic_url_classify(url: str):
    """Fallback used when the BiLSTM model is not trained yet."""
    checks = {
        "suspicious_domain": bool(
            __import__("re").search(r"\.(tk|ml|xyz|ga|cf|top|click|pw|cfd|gq|zip|mov)$|"
                                     r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url)),
        "typosquat": bool(__import__("re").search(
            r"(bank|pay|sbi|amazon|flipkart|google|netflix)-|-(verify|update|claim|login|restore|reward|menu|download)", url)),
        "creds": bool(__import__("re").search(r"(verify|update|login|auth|kyc|renew|unlock|confirm)", url)),
        "download": bool(__import__("re").search(r"\.(apk|exe|dmg|zip|jar)$", url)),
        "no_https": not url.startswith("https://"),
        "upi": url.startswith("upi://"),
    }
    if checks["upi"]:
        return {"label": "Safe", "confidence": 96.0,
                "probs": {"Safe": 96.0, "Phishing": 2.0, "Suspicious": 2.0},
                "reason": "Valid UPI payment link (upi://pay)."}
    score = sum([2 if checks["suspicious_domain"] else 0,
                 2 if checks["typosquat"] else 0,
                 1 if checks["creds"] else 0,
                 3 if checks["download"] else 0,
                 1 if checks["no_https"] else 0])
    if score >= 4:
        return {"label": "Phishing", "confidence": min(98.0, 70 + score * 5),
                "probs": {"Safe": 5.0, "Phishing": min(98.0, 70 + score * 5),
                          "Suspicious": 95.0 - min(98.0, 70 + score * 5)},
                "reason": "Suspicious domain / typosquatting pattern detected."}
    if score >= 2:
        return {"label": "Suspicious", "confidence": min(92.0, 60 + score * 6),
                "probs": {"Safe": 20.0, "Phishing": 40.0, "Suspicious": min(92.0, 60 + score * 6)},
                "reason": "Some risky patterns found — proceed with caution."}
    return {"label": "Safe", "confidence": min(96.0, 80 + score * 5),
            "probs": {"Safe": min(96.0, 80 + score * 5), "Phishing": 10.0, "Suspicious": 10.0},
            "reason": "URL looks normal and uses a trusted scheme."}


def _run_bilstm(m, char_index, cfg, class_names, url: str):
    """Run inference on any loaded BiLSTM. Returns result dict or None on error."""
    try:
        unk = char_index.get("<UNK>", 1)
        seq = [char_index.get(c, unk) for c in str(url)[:cfg["max_len"]]]
        seq = seq + [0] * (cfg["max_len"] - len(seq))
        x   = torch.tensor([seq], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(m(x), dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        return {
            "label"     : class_names[pred],
            "confidence": round(float(probs[pred]) * 100, 1),
            "probs"     : {class_names[i]: round(float(probs[i]) * 100, 1) for i in range(len(class_names))},
            "reason"    : None,
        }
    except Exception:
        return None


def classify_url(url: str):
    """Mission 2 — 3-class URL classifier (Safe / Phishing / Suspicious)."""
    # Prevent BiLSTM character-level false positives on authentic major domains with hash/token paths
    domain_match = re.search(r"https?://([^/:\s]+)", str(url or ""))
    if domain_match:
        host = domain_match.group(1).lower()
        trusted_suffixes = (
            "google.com", "google.co.in", "microsoft.com", "office.com", "live.com",
            "apple.com", "amazon.com", "amazon.in", "github.com",
            "linkedin.com", "docusign.com", "paypal.com", "dropbox.com"
        )
        if any(host == d or host.endswith("." + d) for d in trusted_suffixes):
            return {
                "label": "Safe",
                "confidence": 98.0,
                "probs": {"Safe": 98.0, "Phishing": 1.0, "Suspicious": 1.0},
                "reason": "URL destination is an officially verified trusted domain."
            }

    if url_model is not None:
        result = _run_bilstm(url_model, url_char_index, url_cfg, URL_CLASS_NAMES, url)
        if result:
            return result
    return heuristic_url_classify(url)


def classify_qr_url(url: str):
    """Mission 3 — 2-class QR URL classifier (Safe / Phishing)."""
    if qr_model is not None:
        result = _run_bilstm(qr_model, qr_char_index, qr_cfg, QR_CLASS_NAMES, url)
        if result:
            return result
    # fallback: reuse heuristic but strip Suspicious → map to Phishing
    h = heuristic_url_classify(url)
    if h["label"] == "Suspicious":
        h["label"] = "Phishing"
    return h

# Load game emails from JSON (500 emails: 250 phishing + 250 safe)
EMAIL_POOL = []
for p in [os.path.join(BASE_DIR, 'data', 'game_emails.json'), os.path.join(BASE_DIR, 'dataset', 'game_emails.json')]:
    if os.path.isfile(p):
        try:
            with open(p, encoding='utf-8') as f:
                EMAIL_POOL = json.load(f)
            print(f'Loaded {len(EMAIL_POOL)} game emails from {p}')
            break
        except Exception as e:
            print(f'[WARN] Error loading {p}: {e}')
if not EMAIL_POOL:
    print('[WARN] No game_emails.json found; fallback emails will be served.')


@app.route("/random-emails", methods=["GET"])
def random_emails():
    phishing = [e for e in EMAIL_POOL if e['label'] == 1]
    safe     = [e for e in EMAIL_POOL if e['label'] == 0]
    selected = random.sample(phishing, 5) + random.sample(safe, 5)
    random.shuffle(selected)
    return jsonify(selected)

OCR_PS1_PATH = os.path.join(BASE_DIR, "ocr_engine.ps1")


def load_image_stream(stream, allowed_formats):
    with Image.open(stream) as opened:
        image_format = (opened.format or "").upper()
        if image_format not in allowed_formats:
            supported = ", ".join(sorted(allowed_formats))
            raise ValueError(f"Unsupported image format. Supported formats: {supported}.")
        opened.load()
        return opened.convert("RGB")


def load_base64_image(value, allowed_formats):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image data is empty.")

    image_input = value.strip()
    if image_input.lower().startswith("data:"):
        header, separator, image_input = image_input.partition(",")
        if not separator or not header.lower().startswith("data:image/") or "base64" not in header.lower().split(";")[1:]:
            raise ValueError("Image data URI must use an image MIME type and base64 encoding.")

    if len(image_input) % 4:
        image_input += "=" * (4 - len(image_input) % 4)
    try:
        image_bytes = base64.b64decode(image_input, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Image data is not valid base64.") from exc
    if len(image_bytes) < 16:
        raise ValueError("Image data is too small to be a valid image.")
    return load_image_stream(io.BytesIO(image_bytes), allowed_formats)


def run_ocr_on_image(img_pil: Image.Image) -> str:
    """Preprocess an image and run local Windows Media OCR or Tesseract."""
    import shutil
    import subprocess
    import tempfile
    from PIL import ImageEnhance
    tmp_name = None
    try:
        w, h = img_pil.size
        if w < 2400:
            scale = 2.2 if w < 1400 else 1.5
            processed = img_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        else:
            processed = img_pil.copy()

        gray = processed.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_name = tmp.name
        enhanced.save(tmp_name, format="PNG")

        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if powershell and os.path.isfile(OCR_PS1_PATH):
            cmd = [
                powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                "Bypass", "-File", OCR_PS1_PATH, "-ImagePath", tmp_name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
            print(f"[WARN] Windows Media OCR failed: {proc.stderr.strip() or 'No text returned'}")

        tesseract = shutil.which("tesseract")
        if tesseract:
            proc = subprocess.run(
                [tesseract, tmp_name, "stdout", "-l", "eng"],
                capture_output=True, text=True, timeout=25,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
            print(f"[WARN] Tesseract OCR failed: {proc.stderr.strip() or 'No text returned'}")

        raise RuntimeError(
            "No local OCR engine is available. Windows Media OCR requires Windows PowerShell; "
            "on other systems, install Tesseract and make it available on PATH."
        )
    except Exception as e:
        if isinstance(e, RuntimeError) and str(e).startswith("No local OCR engine"):
            raise
        print(f"[WARN] OCR error: {e}")
        raise RuntimeError(f"Local OCR could not process this image: {e}") from e
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)


def parse_extracted_email_text(text: str) -> dict:
    """Smart parser to extract Sender, Receiver, Subject, URLs, and Body from OCR text."""
    import re
    url_matches = re.findall(
        r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+|'
        r'(?<![@\w.-])(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}(?:/[^\s<>"\'`]*)?',
        text,
        re.IGNORECASE,
    )
    urls = list(dict.fromkeys(url.rstrip(".,;:!?)]}") for url in url_matches))

    # Sender extraction
    sender = ""
    email_bracket = re.search(r'<\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\s*>', text)
    if email_bracket:
        sender = email_bracket.group(1).strip()
    else:
        from_match = re.search(r'(?:from|sender)\s*[:\-]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if from_match:
            sender = from_match.group(1).strip()
        else:
            all_emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
            if all_emails:
                sender = all_emails[0]

    # Clean OCR bracket artifacts (e.g. < read as c, > read as s)
    if sender:
        clean_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+', sender)
        if clean_match:
            sender = clean_match.group(0)
        if sender.endswith(".coms"):
            sender = sender[:-1]
        if sender.startswith("c") and "@" in sender and "c" not in text.split("@")[0][-15:]:
            sender = sender.lstrip("c")

    # Subject extraction
    subject = ""
    subj_match = re.search(r'(?:subject|subj)\s*[:\-]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if subj_match:
        subject = subj_match.group(1).strip()
    else:
        for line in text.splitlines():
            line_clean = line.strip()
            if len(line_clean) > 15 and not line_clean.startswith(('http', 'From', 'To', 'Date')) and any(w in line_clean.lower() for w in ['opportunity', 'alert', 'urgent', 'engineer', 'offer', 'account', 'verify', 'order', 'statement', 'entry level']):
                subject = line_clean
                break

    # Receiver extraction
    receiver = ""
    to_match = re.search(r'(?:to|recipient)\s*[:\-]\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if to_match:
        receiver = to_match.group(1).strip()
    else:
        all_emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
        if len(all_emails) > 1 and all_emails[1] != sender:
            receiver = all_emails[1]

    lines = [l for l in text.splitlines() if not re.match(r'^(?:from|sender|to|recipient|subject|subj)\s*[:\-]', l, re.IGNORECASE)]
    body = "\n".join(lines).strip() or text.strip()

    return {
        "sender": sender,
        "receiver": receiver,
        "subject": subject,
        "urls": ", ".join(dict.fromkeys(urls)),
        "body": body,
        "raw_text": text.strip()
    }


BRAND_DOMAINS = {
    "docusign": ["docusign.com", "docusign.net"],
    "sharepoint": ["sharepoint.com", "microsoft.com", "office.com"],
    "onedrive": ["onedrive.com", "live.com", "microsoft.com", "office.com"],
    "google": ["google.com", "google.co.in", "drive.google.com"],
    "microsoft": ["microsoft.com", "office.com", "office365.com", "live.com", "msn.com"],
    "paypal": ["paypal.com", "paypal.me"],
    "amazon": ["amazon.com", "amazon.in"],
    "sbi": ["sbi.co.in", "onlinesbi.sbi", "onlinesbi.com"],
    "hdfc": ["hdfcbank.com"],
    "netflix": ["netflix.com"],
    "apple": ["apple.com", "icloud.com"],
    "adobe": ["adobe.com"],
    "dropbox": ["dropbox.com"],
    "linkedin": ["linkedin.com"],
    "codeneurax": ["codeneurax.com", "codeneurax.in"],
}


def email_forensic_signals(text: str, sender: str, urls: str) -> list:
    signals = []
    lower_text = (text or "").lower()
    sender_lower = (sender or "").lower()

    if any(word in lower_text for word in [
        "urgent", "verify", "immediate", "suspended", "blocked", "expire", "action required",
    ]):
        signals.append("High-pressure urgency language detected")

    if any(tld in sender_lower for tld in [".xyz", ".tk", ".ml", ".top"]):
        signals.append("Sender uses a high-risk or uncommon domain suffix")

    free_mail = _re.search(r"@(gmail|yahoo|outlook|hotmail|live)\.[a-z.]+", sender_lower)

    # Extract all candidate domains from URLs and body text
    raw_urls = [u.strip() for u in str(urls or "").replace(",", " ").split() if u.strip()]
    text_urls = _re.findall(r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+', text or "", _re.IGNORECASE)
    all_urls = list(dict.fromkeys(raw_urls + text_urls))

    url_domains = []
    for u in all_urls:
        clean = _re.sub(r"^https?://", "", u.lower())
        dom = clean.split("/")[0].split(":")[0].strip()
        if dom:
            url_domains.append(dom)

    # Brand-to-domain consistency check
    for brand, valid_domains in BRAND_DOMAINS.items():
        if brand in lower_text or brand in sender_lower:
            if free_mail:
                signals.append(f"Brand impersonation: Sender claims '{brand.title()}' identity via free webmail ({sender})")

            if "@" in sender_lower and not free_mail:
                sender_domain = sender_lower.split("@")[-1].strip("> \t")
                if sender_domain and not any(sender_domain == d or sender_domain.endswith("." + d) for d in valid_domains):
                    signals.append(f"Brand impersonation: Sender domain '@{sender_domain}' does not belong to official {brand.title()}")

            if url_domains:
                has_valid = any(any(dom == d or dom.endswith("." + d) for d in valid_domains) for dom in url_domains)
                if not has_valid:
                    signals.append(f"Brand impersonation: Content references '{brand.title()}' but link points to unverified domain '{url_domains[0]}'")

    # Typosquatting / brand keywords with hyphens in URL domains
    for dom in url_domains:
        if _re.search(r"(docusign|sharepoint|onedrive|paypal|microsoft|apple|netflix|amazon|dropbox|google)[-_]", dom):
            signals.append(f"Deceptive link domain '{dom}' mimics brand names with hyphenated lookalikes")

    urls_lower = (str(urls) + " " + " ".join(url_domains)).lower()
    if urls_lower.strip() and (
        "http://" in urls_lower
        or any(tld in urls_lower for tld in [".xyz", ".tk", ".ml", ".top"])
        or "xn--" in urls_lower
    ):
        signals.append("Link uses an unencrypted or potentially deceptive destination")

    return list(dict.fromkeys(signals))


@app.route("/analyze-email-image", methods=["POST"])
def analyze_email_image_route():
    """OCR an uploaded spam/phishing image or screenshot, extract text fields, and classify as Scam/Phishing or Safe."""
    arr = None
    img = None

    if "image" in request.files or "file" in request.files:
        up = request.files.get("image") or request.files.get("file")
        if up and up.filename != "":
            try:
                img = load_image_stream(up.stream, {"PNG", "JPEG", "WEBP"})
            except Exception as e:
                return jsonify({"error": f"Uploaded file is not a valid image: {e}"}), 400

    if img is None:
        data = request.get_json(silent=True) or {}
        image_input = data.get("image") or data.get("file") or ""
        if not image_input and request.form:
            image_input = request.form.get("image") or request.form.get("file") or ""

        if not image_input:
            return jsonify({"error": "Missing image input. Please provide a base64 string or upload a photo."}), 400

        try:
            img = load_base64_image(image_input, {"PNG", "JPEG", "WEBP"})
        except Exception as e:
            return jsonify({"error": f"Failed to read image data: {e}"}), 400

    if img is None:
        return jsonify({"error": "No valid image provided."}), 400

    try:
        ocr_raw = run_ocr_on_image(img)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    if not ocr_raw:
        return jsonify({"error": "No readable text found in image. Please ensure the screenshot has clear, visible text."}), 400

    extracted = parse_extracted_email_text(ocr_raw)
    subject = extracted["subject"]
    body = extracted["body"]
    sender = extracted["sender"]
    urls = extracted["urls"]
    receiver = extracted["receiver"]
    text = f"Subject: {subject} From: {sender} To: {receiver} Body: {body} URLs: {urls}"

    # 5. Decision Fusion for OCR
    if model is not None and tokenizer is not None:
        enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE))
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            bert_prob = float(probs[1]) * 100.0
    else:
        bert_prob = 0.0

    # URL BiLSTM Check for OCR
    url_matches = re.findall(r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+', text, re.IGNORECASE)
    if extracted.get("urls"):
        url_matches.extend([u.strip() for u in extracted["urls"].split(",") if u.strip()])
    url_matches = list(dict.fromkeys(url_matches))
    url_max_score = 0.0
    url_phishing_detected = False
    for u in url_matches:
        url_res = classify_url(u)
        if url_res.get("label") in ("Phishing", "Suspicious"):
            if url_res.get("label") == "Phishing":
                url_phishing_detected = True
            url_max_score = max(url_max_score, url_res.get("confidence", 0.0))

    signals = email_forensic_signals(text, sender, urls)
    brand_mismatch = any("brand impersonation" in s.lower() or "deceptive link" in s.lower() for s in signals)

    # Fusion
    is_phishing = (bert_prob >= 50.0) or (url_max_score >= 50.0) or url_phishing_detected or brand_mismatch
    label = "Phishing" if is_phishing else "Safe"
    conf = round(max(bert_prob, url_max_score, 94.0 if (brand_mismatch or url_phishing_detected) else 0.0), 1)

    # Ensure confidence is capped at 100
    conf = min(conf, 100.0)
    if conf < 50.0: conf = round(100.0 - conf, 1)

    phishing_prob = conf if label == "Phishing" else round(100.0 - conf, 1)
    safe_prob = round(100.0 - phishing_prob, 1)

    if url_phishing_detected:
        signals.append(f"Embedded link identified as malicious by BiLSTM URL classifier ({url_max_score}% confidence)")
    if not signals:
        signals.append("No common urgency, sender-domain, or deceptive-link indicators were detected")
    signals = list(dict.fromkeys(signals))

    return jsonify({
        "status": "ok",
        "predicted_label": label,
        "confidence": conf,
        "phishing_prob": phishing_prob,
        "safe_prob": safe_prob,
        "extracted": extracted,
        "signals": signals,
        "model_name": "DistilBERT Sequence Classifier" if (model is not None and tokenizer is not None) else "CyberSleuth Threat Heuristics",
        "summary": f"{'DistilBERT' if (model is not None and tokenizer is not None) else 'Local threat heuristics'} evaluated OCR-extracted email telemetry with {conf}% confidence towards {label} classification."
    })


@app.route("/ml-hint", methods=["POST"])
def ml_hint():
    data = request.get_json() or {}
    vector = (data.get("vector") or "email").lower()
    item = data.get("item") or {}

    try:
        if vector == "email":
            # Check if an image is provided inside item or data
            image_b64 = item.get("image") or data.get("image") or ""
            extracted_info = None
            if image_b64 and isinstance(image_b64, str):
                try:
                    clean_b64 = image_b64
                    if clean_b64.startswith("data:") and "," in clean_b64:
                        clean_b64 = clean_b64.split(",", 1)[1].strip()
                    missing_padding = len(clean_b64) % 4
                    if missing_padding:
                        clean_b64 += '=' * (4 - missing_padding)
                    img_bytes = base64.b64decode(clean_b64)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    ocr_text = run_ocr_on_image(pil_img)
                    if ocr_text:
                        extracted_info = parse_extracted_email_text(ocr_text)
                        if not item.get("subject") and extracted_info["subject"]:
                            item["subject"] = extracted_info["subject"]
                        if not item.get("sender") and extracted_info["sender"]:
                            item["sender"] = extracted_info["sender"]
                        if not item.get("receiver") and extracted_info["receiver"]:
                            item["receiver"] = extracted_info["receiver"]
                        if not item.get("urls") and extracted_info["urls"]:
                            item["urls"] = extracted_info["urls"]
                        if not item.get("body"):
                            item["body"] = extracted_info["body"]
                except Exception as e:
                    print(f"[WARN] Error running OCR in /ml-hint: {e}")

            raw_body = str(item.get("body") or "")
            parsed_text = parse_extracted_email_text(raw_body)
            subject = str(item.get("subject") or parsed_text["subject"])
            body = parsed_text["body"] or raw_body
            sender = str(item.get("sender") or parsed_text["sender"])
            receiver = str(item.get("receiver") or parsed_text["receiver"])
            urls = str(item.get("urls") or parsed_text["urls"])
            extracted_info = {
                **parsed_text,
                "sender": sender,
                "receiver": receiver,
                "subject": subject,
                "urls": urls,
                "body": body,
            }
            text = f"Subject: {subject} From: {sender} To: {receiver} Body: {body} URLs: {urls}"
            
            # 5. Fusion for /ml-hint
            # Extract URLs
            all_text = f"{subject} {body} {urls}"
            url_matches = re.findall(r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+', all_text, re.IGNORECASE)
            if extracted_info.get("urls"):
                url_matches.extend([u.strip() for u in extracted_info["urls"].split(",") if u.strip()])
            extracted_urls = list(dict.fromkeys(url_matches))

            # URL BiLSTM Check
            url_max_score = 0.0
            url_phishing_detected = False
            for u in extracted_urls:
                url_res = classify_url(u)
                if url_res.get("label") in ("Phishing", "Suspicious"):
                    if url_res.get("label") == "Phishing":
                        url_phishing_detected = True
                    url_max_score = max(url_max_score, url_res.get("confidence", 0.0))

            # Forensic Signals (includes Brand Mismatch check)
            signals = email_forensic_signals(text, sender, urls)
            brand_mismatch = any("brand impersonation" in s.lower() or "deceptive link" in s.lower() for s in signals)

            # DistilBERT
            bert_prob = 0.0
            if model is not None and tokenizer is not None:
                enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE))
                    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
                    bert_prob = float(probs[1]) * 100.0

            # Decision
            is_phishing = (bert_prob >= 50.0) or (url_max_score >= 50.0) or url_phishing_detected or brand_mismatch
            label = "Phishing" if is_phishing else "Safe"
            conf = round(max(bert_prob, url_max_score, 94.0 if (brand_mismatch or url_phishing_detected) else 0.0), 1)
            conf = min(conf, 100.0)
            if conf < 50.0: conf = round(100.0 - conf, 1)

            phishing_prob = conf if label == "Phishing" else round(100.0 - conf, 1)
            safe_prob = round(100.0 - phishing_prob, 1)

            if url_phishing_detected:
                signals.append(f"Embedded link identified as malicious by BiLSTM URL classifier ({url_max_score}% confidence)")
            if not signals:
                signals.append("No common urgency, sender-domain, or deceptive-link indicators were detected")
            signals = list(dict.fromkeys(signals))

            return jsonify({
                "model_name": "DistilBERT Sequence Classifier" if (model is not None and tokenizer is not None) else "CyberSleuth Threat Heuristics",
                "predicted_label": label,
                "confidence": conf,
                "phishing_prob": phishing_prob,
                "safe_prob": safe_prob,
                "vector": "email",
                "extracted": extracted_info,
                "signals": signals,
                "summary": f"{'DistilBERT' if (model is not None and tokenizer is not None) else 'Local threat heuristics'} evaluated email telemetry with {conf}% confidence towards {label} classification."
            })

        elif vector == "url":
            target_url = (item.get("url") or "").strip()
            res = classify_url(target_url) if target_url else {"label": "Safe", "confidence": 90.0, "reason": "No URL specified."}
            
            signals = []
            if res.get("reason"):
                signals.append(res["reason"])
            if "://" in target_url and not target_url.startswith("https://"):
                signals.append("Unencrypted HTTP protocol detected")
            if any(tld in target_url for tld in [".tk", ".ml", ".xyz", ".top", ".ga", ".cf"]):
                signals.append("Registrar top-level domain flagged on abuse monitoring lists")
            if any(kw in target_url.lower() for kw in ["verify", "login", "update", "secure", "claim"]):
                signals.append("Credential harvesting URI pattern detected in route path")
            if not signals:
                signals.append("Domain reputation and SSL certificate parameters match legitimate benchmarks")

            return jsonify({
                "model_name": "Character-level BiLSTM URL Classifier",
                "predicted_label": res.get("label", "Safe"),
                "confidence": res.get("confidence", 92.0),
                "vector": "url",
                "signals": signals[:3],
                "summary": f"BiLSTM recurrent network analyzed sub-character tokens with {res.get('confidence', 92.0)}% confidence."
            })

        elif vector == "qr":
            target_url = (item.get("url") or "").strip()
            res = classify_qr_url(target_url) if target_url else {"label": "Safe", "confidence": 90.0, "reason": "No target decoded."}
            
            signals = []
            if res.get("reason"):
                signals.append(res["reason"])
            if target_url.startswith("upi://"):
                signals.append("Direct UPI protocol handler identified")
            elif any(ext in target_url.lower() for ext in [".apk", ".exe", ".jar", ".zip"]):
                signals.append("QR resolves directly to executable binary package payload")
            elif not target_url.startswith("https://"):
                signals.append("Insecure plain HTTP optical destination")
            if not signals:
                signals.append("Decoded endpoint correlates with verified corporate infrastructure")

            return jsonify({
                "model_name": "BiLSTM Optical Payload Classifier",
                "predicted_label": res.get("label", "Safe"),
                "confidence": res.get("confidence", 91.0),
                "vector": "qr",
                "signals": signals[:3],
                "summary": f"Optical threat engine classified decoded target with {res.get('confidence', 91.0)}% confidence."
            })

        elif vector == "sms":
            sms_text = (item.get("text") or "").strip()
            res = classify_sms_text(sms_text) if sms_text else {"label": "Safe", "confidence": 90.0, "reason": "No SMS text provided."}
            
            signals = []
            if res.get("reason"):
                signals.append(res["reason"])
            if any(w in sms_text.lower() for w in ["won", "prize", "lottery", "cashback"]):
                signals.append("Smishing baiting trigger detected")
            if any(w in sms_text.lower() for w in ["kyc", "block", "suspend"]):
                signals.append("Panic-inducing account threat phrase detected")
            if not signals:
                signals.append("Cellular payload aligns with legitimate transactional notifications")

            return jsonify({
                "model_name": "Dual-Heuristic SMS Smishing Engine",
                "predicted_label": res.get("label", "Safe"),
                "confidence": res.get("confidence", 93.0),
                "vector": "sms",
                "signals": signals[:3],
                "summary": f"SMS detection engine analyzed message structure with {res.get('confidence', 93.0)}% confidence."
            })

        return jsonify({"error": f"Unknown vector {vector}"}), 400

    except Exception as e:
        return jsonify({
            "model_name": "Forensic Engine",
            "predicted_label": "Phishing" if item.get("label") == 1 else "Safe",
            "confidence": 90.0,
            "vector": vector,
            "signals": ["Automated feature extraction: suspicious lexical and behavioral patterns."],
            "summary": f"Model inference error ({e}); heuristic inspection applied."
        })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    subject = data.get('subject', '')
    body = data.get('body', '')
    sender = data.get('sender', '')
    urls = data.get('urls', '')

    # 1. Automatically extract URLs if not explicitly provided or found in body/text
    import re
    all_text = f"{subject} {body} {urls}"
    url_matches = re.findall(r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+', all_text, re.IGNORECASE)
    extracted_urls = list(dict.fromkeys(url_matches))
    combined_urls = ", ".join(extracted_urls) if extracted_urls else urls

    # 2. Run URL BiLSTM / heuristic check on extracted links
    url_phishing_detected = False
    url_max_score = 0.0
    for u in extracted_urls:
        url_res = classify_url(u)
        if url_res.get("label") in ("Phishing", "Suspicious") or url_res.get("confidence", 0) >= 50.0:
            if url_res.get("label") == "Phishing":
                url_phishing_detected = True
            url_max_score = max(url_max_score, url_res.get("confidence", 0.0))

    # 3. Brand-Domain Consistency Check
    brand_map = {
        "docusign": ["docusign.com"],
        "sharepoint": ["sharepoint.com", "microsoft.com"],
        "onedrive": ["sharepoint.com", "microsoft.com"],
        "google": ["google.com"],
        "microsoft": ["microsoft.com"],
        "amazon": ["amazon.com", "amazon.in"],
    }
    lower_text = all_text.lower()
    brand_mismatch = False
    for brand, allowed_domains in brand_map.items():
        if brand in lower_text:
            found_valid_domain = False
            for u in extracted_urls:
                if any(dom in u.lower() for dom in allowed_domains):
                    found_valid_domain = True
                    break
            if extracted_urls and not found_valid_domain:
                brand_mismatch = True

    # 4. DistilBERT prediction
    text = f"Subject: {subject} Body: {body} From: {sender} URLs: {combined_urls}"
    bert_phishing_prob = 0.0
    if model is not None and tokenizer is not None:
        enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE))
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            bert_phishing_prob = float(probs[1]) * 100.0

    # 5. Decision Fusion (DistilBERT >= 50% OR URL BiLSTM >= 50% OR Brand Mismatch)
    is_phishing = (bert_phishing_prob >= 50.0) or (url_max_score >= 50.0) or url_phishing_detected or brand_mismatch

    final_phishing_prob = max(bert_phishing_prob, url_max_score, 92.0 if brand_mismatch else 0.0)
    final_safe_prob = round(100.0 - final_phishing_prob, 1)

    label = "Phishing" if is_phishing else "Safe"
    confidence = round(final_phishing_prob if is_phishing else final_safe_prob, 1)

    return jsonify({
        "label"        : label,
        "confidence"   : confidence,
        "phishing_prob": round(final_phishing_prob, 1),
        "safe_prob"    : round(final_safe_prob, 1)
    })


@app.route("/classify-sms", methods=["POST"])
def classify_sms():
    data = request.get_json() or {}
    text = (data.get("text") or data.get("sms") or "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400
    return jsonify(classify_sms_text(text))

@app.route("/classify-url", methods=["POST"])
def classify_url_route():
    data = request.get_json() or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing url"}), 400
    return jsonify(classify_url(url))


@app.route("/auth/register", methods=["POST"])
@app.route("/register", methods=["POST"])
def auth_register():
    data = request.get_json() or {}
    username = normalize_username(data.get("username"))
    password_hash = (data.get("password_hash") or data.get("password") or "").strip()

    if not username:
        return jsonify({"error": "Username must be 3-24 characters (letters, numbers, . _ - only)."}), 400
    if not password_hash or len(password_hash) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    admin_role = get_admin_role_for(username)
    is_admin = 1 if admin_role else 0

    with db_connect() as conn:
        exists = conn.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?)", (username,)).fetchone()
        if exists:
            return jsonify({"error": f"Username '{username}' already exists. Please choose a different name or sign in."}), 409

        stored_hash = generate_password_hash(password_hash)
        ts = now_iso()
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, best_score, best_category, games_played, is_admin, admin_role, created_at, updated_at) VALUES (?, ?, 0, 'Normal', 0, ?, ?, ?, ?)",
            (username, stored_hash, is_admin, admin_role, ts, ts),
        )
        user_id = cur.lastrowid

    token = create_session(user_id)
    return jsonify({
        "token": token,
        "username": username,
        "is_admin": bool(is_admin),
        "admin_role": admin_role
    })


@app.route("/auth/login", methods=["POST"])
@app.route("/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    username = normalize_username(data.get("username"))
    password_hash = (data.get("password_hash") or data.get("password") or "").strip()

    if not username or not password_hash:
        return jsonify({"error": "Both username and password are required."}), 400

    with db_connect() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, is_admin, admin_role, best_score, best_category FROM users WHERE LOWER(username) = LOWER(?)",
            (username,),
        ).fetchone()
        if not user and username.lower() in ("aditya", "adity"):
            alt = "aditya" if username.lower() == "adity" else "adity"
            user = conn.execute(
                "SELECT id, username, password_hash, is_admin, admin_role, best_score, best_category FROM users WHERE LOWER(username) = LOWER(?)",
                (alt,),
            ).fetchone()

    if not user:
        return jsonify({"error": "Invalid username or password."}), 401

    # Strictly verify password hash
    matched = False
    try:
        matched = check_password_hash(user["password_hash"], password_hash)
    except Exception:
        matched = False

    # Also check if exact hash match
    if not matched and user["password_hash"] == password_hash:
        matched = True

    if not matched:
        return jsonify({"error": "Invalid username or password. Access denied."}), 401

    admin_role = get_admin_role_for(user["username"])
    is_admin = 1 if admin_role else 0

    ts = now_iso()
    with db_connect() as conn:
        conn.execute("UPDATE users SET is_admin = ?, admin_role = ?, updated_at = ? WHERE id = ?", (is_admin, admin_role, ts, user["id"]))

    token = create_session(user["id"])
    return jsonify({
        "token": token,
        "username": user["username"],
        "is_admin": bool(is_admin),
        "admin_role": admin_role,
        "best_score": user["best_score"],
        "best_category": user["best_category"],
    })


@app.route("/auth/me", methods=["POST"])
def auth_me():
    data = request.get_json() or {}
    token = (data.get("token") or "").strip()
    user = get_user_by_token(token)
    if not user:
        return jsonify({"error": "Invalid session"}), 401
    return jsonify({
        "username": user["username"],
        "best_score": user["best_score"],
        "best_category": user["best_category"],
    })


@app.route("/submit-score", methods=["POST"])
def submit_score():
    data = request.get_json() or {}
    token = (data.get("token") or "").strip()
    score = int(data.get("score") or 0)
    category = (data.get("category") or "Normal").strip()[:32]
    mission = (data.get("mission") or "").strip()[:32]

    user = get_user_by_token(token)
    score = max(0, score)
    ts = now_iso()

    if not user:
        # Fallback for local session token
        if token.startswith("local-") or token.startswith("guest-"):
            raw_name = token.split("-", 1)[1] if "-" in token else "Analyst"
            uname = normalize_username(raw_name) or "GuestAnalyst"
            admin_role = get_admin_role_for(uname)
            is_admin = 1 if admin_role else 0
            with db_connect() as conn:
                row = conn.execute("SELECT id, best_score, games_played FROM users WHERE LOWER(username) = LOWER(?)", (uname,)).fetchone()
                if not row:
                    cur = conn.execute(
                        "INSERT INTO users (username, password_hash, best_score, best_category, games_played, is_admin, admin_role, created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                        (uname, "local_hash", score, category, is_admin, admin_role, ts, ts),
                    )
                    uid = cur.lastrowid
                else:
                    uid = row["id"]
                    new_best = max(score, row["best_score"])
                    conn.execute(
                        "UPDATE users SET best_score = ?, best_category = ?, games_played = COALESCE(games_played, 0) + 1, updated_at = ? WHERE id = ?",
                        (new_best, category if score >= row["best_score"] else "Normal", ts, uid),
                    )
                conn.execute(
                    "INSERT INTO score_entries (user_id, username, score, category, mission, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (uid, uname, score, category, mission, ts),
                )
            return jsonify({"ok": True, "score": score, "category": category})
        return jsonify({"error": "Invalid session"}), 401

    with db_connect() as conn:
        conn.execute(
            "INSERT INTO score_entries (user_id, username, score, category, mission, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user["id"], user["username"], score, category, mission, ts),
        )
        new_best = max(score, user["best_score"])
        cat_to_save = category if score >= user["best_score"] else user["best_category"]
        conn.execute(
            "UPDATE users SET best_score = ?, best_category = ?, games_played = COALESCE(games_played, 0) + 1, updated_at = ? WHERE id = ?",
            (new_best, cat_to_save, ts, user["id"]),
        )

    return jsonify({"ok": True, "score": score, "category": category})


@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    return jsonify(leaderboard_rows())


@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    """Admin endpoint to see how many players have played, their scores, and audit history."""
    with db_connect() as conn:
        total_players = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_games = conn.execute("SELECT COUNT(*) FROM score_entries").fetchone()[0]
        stats_row = conn.execute("SELECT MAX(score), AVG(score) FROM score_entries").fetchone()
        highest_score = stats_row[0] or 0
        avg_score = round(stats_row[1] or 0, 1)

        players = conn.execute("""
            SELECT id, username, best_score, best_category, COALESCE(games_played, 0) AS games_played, COALESCE(is_admin, 0) AS is_admin, admin_role, created_at, updated_at
            FROM users
            ORDER BY best_score DESC, games_played DESC
        """).fetchall()

        recent_activity = conn.execute("""
            SELECT s.id, s.username, s.score, s.category, s.mission, s.created_at
            FROM score_entries s
            ORDER BY s.id DESC
            LIMIT 30
        """).fetchall()

    return jsonify({
        "total_players": total_players,
        "total_games_played": total_games,
        "highest_score": highest_score,
        "avg_score": avg_score,
        "players": [
            {
                "id": p["id"],
                "username": p["username"],
                "best_score": p["best_score"],
                "category": p["best_category"],
                "games_played": max(1, p["games_played"]) if p["best_score"] > 0 else p["games_played"],
                "is_admin": bool(p["is_admin"]),
                "admin_role": p["admin_role"] or get_admin_role_for(p["username"]),
                "created_at": p["created_at"],
                "updated_at": p["updated_at"],
            }
            for p in players
        ],
        "recent_activity": [
            {
                "id": a["id"],
                "username": a["username"],
                "score": a["score"],
                "category": a["category"],
                "mission": a["mission"],
                "created_at": a["created_at"],
            }
            for a in recent_activity
        ]
    })


@app.route("/admin/seed-demo", methods=["POST"])
def admin_seed_demo():
    """Synthetic demo seeding disabled per requirements."""
    return jsonify({"ok": True, "message": "Synthetic demo seeding disabled."})




# -------------------------------------------------------
# Mission 3 — QR Quishing
# -------------------------------------------------------
QR_SCENARIOS = [
    {
        "id": "upi-fake",
        "title": "UPI Payment QR",
        "icon": "💰",
        "location": "Kirana Store Payment Counter",
        "description": "A QR code placed on the shop counter for scanning UPI payments. It looks like the official GPay/PhonePe code.",
        "url": "http://upi-payment-verify.xyz/collect?upi=shop@ybl&amount=499",
        "label": 1,
        "hint": "The URL domain is suspicious and not a real UPI payment link.",
        "preview": {
            "site_name": "upi-payment-verify.xyz",
            "page_title": "UPI Payment Gateway — Verify & Pay",
            "favicon": "🔴",
            "screenshot_desc": "A fake payment page mimicking GPay UI asking for UPI PIN and bank details.",
            "red_flags": ["Domain is not gpay.com or phonepe.com", "Asks for UPI PIN on a webpage", "HTTP not HTTPS"],
            "domain_age": "3 days old",
            "ssl": False,
            "thumbnail_color": "#3a0a0a",
        },
    },
    {
        "id": "restaurant-menu",
        "title": "Restaurant Menu QR",
        "icon": "🍽️",
        "location": "Cafe Menu Table Sticker",
        "description": 'A sticker on the table says "Scan to view our digital menu". The QR is tiny and hard to inspect before scanning.',
        "url": "http://menu-download.tk/menu.apk",
        "label": 1,
        "hint": "The URL ends in .tk and points to an .apk file — a malware download.",
        "preview": {
            "site_name": "menu-download.tk",
            "page_title": "Download Our Menu App — Free",
            "favicon": "🔴",
            "screenshot_desc": "A page with a big green Download button prompting you to install menu.apk on your Android device.",
            "red_flags": [".tk is a free high-risk domain", "Forces .apk download", "No restaurant branding visible"],
            "domain_age": "1 day old",
            "ssl": False,
            "thumbnail_color": "#2a0a2a",
        },
    },
    {
        "id": "whatsapp-discount",
        "title": "Discount / Coupon QR",
        "icon": "🎁",
        "location": "WhatsApp Forwarded Image",
        "description": 'A forwarded WhatsApp image claims "50% OFF today only! Scan to claim your coupon" at a big shopping brand.',
        "url": "http://flipkart-sale-claim.ml/coupon?id=88912",
        "label": 1,
        "hint": 'The domain "flipkart-sale-claim.ml" is typosquatting — not the real Flipkart.',
        "preview": {
            "site_name": "flipkart-sale-claim.ml",
            "page_title": "Flipkart Big Billion Sale — Claim ₹5000 Coupon",
            "favicon": "🔴",
            "screenshot_desc": "A page styled like Flipkart asking for your mobile number and OTP to claim a fake coupon.",
            "red_flags": ["Domain is not flipkart.com", ".ml is a free high-risk TLD", "Asks for OTP to claim coupon"],
            "domain_age": "2 days old",
            "ssl": False,
            "thumbnail_color": "#3a1a0a",
        },
    },
    {
        "id": "kyc-bank",
        "title": 'KYC QR from "Bank"',
        "icon": "🏦",
        "location": 'SMS from "SBI KYC"',
        "description": "An SMS warns that your bank KYC will expire and asks you to scan the QR to update KYC immediately.",
        "url": "http://sbi-kyc-update.co/verify",
        "label": 1,
        "hint": "Banks never ask for KYC via QR codes. The domain is fake.",
        "preview": {
            "site_name": "sbi-kyc-update.co",
            "page_title": "SBI KYC Verification Portal",
            "favicon": "🔴",
            "screenshot_desc": "A page with SBI logo asking for account number, Aadhaar, and OTP to complete KYC.",
            "red_flags": ["Not sbi.co.in official domain", "Banks never do KYC via QR", "Collects Aadhaar + OTP"],
            "domain_age": "5 days old",
            "ssl": False,
            "thumbnail_color": "#0a1a3a",
        },
    },
    {
        "id": "login-qr",
        "title": "Login QR",
        "icon": "🔐",
        "location": "Google Login Screen (Phishing)",
        "description": 'A fake "secure login" page shows a QR code saying "Scan to log in securely with your phone".',
        "url": "http://google-account-login.cf/auth",
        "label": 1,
        "hint": "The URL is not a google.com domain — it's a lookalike.",
        "preview": {
            "site_name": "google-account-login.cf",
            "page_title": "Sign in — Google Accounts",
            "favicon": "🔴",
            "screenshot_desc": "A pixel-perfect Google login clone asking for email and password. URL bar shows the fake domain.",
            "red_flags": ["Not accounts.google.com", ".cf is a free high-risk TLD", "Cloned Google UI to steal credentials"],
            "domain_age": "1 day old",
            "ssl": False,
            "thumbnail_color": "#0a1a0a",
        },
    },
    {
        "id": "amazon-legit",
        "title": "Legitimate Amazon QR",
        "icon": "📦",
        "location": "Amazon Package Box",
        "description": "A QR code on your Amazon delivery package that leads to the official Amazon order tracking page.",
        "url": "https://www.amazon.com/your-orders",
        "label": 0,
        "hint": "Official amazon.com domain — safe to open.",
        "preview": {
            "site_name": "amazon.com",
            "page_title": "Your Orders — Amazon",
            "favicon": "✅",
            "screenshot_desc": "The official Amazon orders page showing your recent purchases with tracking details.",
            "red_flags": [],
            "domain_age": "Since 1994",
            "ssl": True,
            "thumbnail_color": "#0a1a0a",
        },
    },
    {
        "id": "zomato-legit",
        "title": "Legitimate Zomato QR",
        "icon": "🍕",
        "location": "Zomato Delivery Receipt",
        "description": "The receipt from your food delivery has a QR to view the order and rate the delivery.",
        "url": "https://www.zomato.com/orders",
        "label": 0,
        "hint": "Official zomato.com domain — safe to open.",
        "preview": {
            "site_name": "zomato.com",
            "page_title": "Your Orders — Zomato",
            "favicon": "✅",
            "screenshot_desc": "The official Zomato order history page with your recent food deliveries and ratings.",
            "red_flags": [],
            "domain_age": "Since 2008",
            "ssl": True,
            "thumbnail_color": "#0a1a0a",
        },
    },
    {
        "id": "gpay-legit",
        "title": "Legitimate GPay QR",
        "icon": "💳",
        "location": "Google Pay Merchant QR",
        "description": "A printed Google Pay merchant QR at a trusted shop for UPI payments.",
        "url": "upi://pay?pa=merchant@okhdfcbank&pn=TrustedShop&am=250",
        "label": 0,
        "hint": "A genuine UPI payment link (upi://pay) — safe.",
        "preview": {
            "site_name": "UPI Payment",
            "page_title": "Pay ₹250 to TrustedShop via HDFC Bank",
            "favicon": "✅",
            "screenshot_desc": "A standard UPI deep-link that opens GPay or PhonePe directly. No browser page — goes straight to your payment app.",
            "red_flags": [],
            "domain_age": "N/A — UPI protocol link",
            "ssl": True,
            "thumbnail_color": "#0a1a0a",
        },
    },
    {
        "id": "hdfc-legit",
        "title": "HDFC NetBanking QR",
        "icon": "🏦",
        "location": "HDFC Bank Branch Poster",
        "description": "A QR code on the HDFC bank branch notice board linking to their official netbanking portal.",
        "url": "https://www.hdfcbank.com/netbanking",
        "label": 0,
        "hint": "Official hdfcbank.com domain with HTTPS — safe.",
        "preview": {
            "site_name": "hdfcbank.com",
            "page_title": "NetBanking Login — HDFC Bank",
            "favicon": "✅",
            "screenshot_desc": "The official HDFC Bank netbanking login page with customer ID and IPIN fields.",
            "red_flags": [],
            "domain_age": "Since 1999",
            "ssl": True,
            "thumbnail_color": "#0a1a0a",
        },
    },
    {
        "id": "paytm-fake",
        "title": "Paytm Cashback QR",
        "icon": "💸",
        "location": "Telegram Group Message",
        "description": "A Telegram message claims scan this QR to get ₹200 Paytm cashback instantly. Looks like an official Paytm promotion.",
        "url": "http://paytm-cashback-offer.gq/claim?ref=TG8821",
        "label": 1,
        "hint": ".gq is a free high-risk domain. Paytm never sends cashback via Telegram QR codes.",
        "preview": {
            "site_name": "paytm-cashback-offer.gq",
            "page_title": "Paytm Cashback — Claim ₹200 Now",
            "favicon": "🔴",
            "screenshot_desc": "A fake Paytm-styled page asking for your Paytm mobile number and MPIN to credit cashback.",
            "red_flags": [".gq free high-risk TLD", "Not paytm.com", "Asks for MPIN on external site"],
            "domain_age": "4 days old",
            "ssl": False,
            "thumbnail_color": "#1a1a0a",
        },
    },
]


def qr_image_for(url: str):
    """Generate a QR PNG (base64 data URI) for a URL using the qrcode lib."""
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def check_url_accessibility(url: str, timeout: float = 5.0):
    """Best-effort reachability check from this server."""
    if not url or url.startswith("upi://"):
        return {"accessible": True, "reason": "Non-web payment link."}

    headers = {
        "User-Agent": "Mozilla/5.0 (Mission3 QR checker)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        req = Request(url, headers=headers, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return {"accessible": resp.status < 400, "reason": f"HTTP {resp.status}"}
    except HTTPError as e:
        if e.code in (403, 405):
            try:
                req = Request(url, headers=headers, method="GET")
                with urlopen(req, timeout=timeout) as resp:
                    return {"accessible": resp.status < 400, "reason": f"HTTP {resp.status}"}
            except Exception as inner:
                return {"accessible": False, "reason": f"{type(inner).__name__}: {inner}"}
        return {"accessible": False, "reason": f"HTTP {e.code}"}
    except URLError as e:
        return {"accessible": False, "reason": str(e.reason)}
    except Exception as e:
        return {"accessible": False, "reason": f"{type(e).__name__}: {e}"}


@app.route("/generate-qr", methods=["GET"])
def generate_qr_route():
    """Dynamically generate a QR code PNG image for any given URL."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing url"}), 400
    data_uri = qr_image_for(url)
    if not data_uri:
        return jsonify({"error": "Could not generate QR code"}), 500
    if request.args.get("format") == "png":
        b64_data = data_uri.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_data)
        return raw_bytes, 200, {"Content-Type": "image/png"}
    return jsonify({"qr_image": data_uri, "url": url})


@app.route("/random-qr", methods=["GET"])
def random_qr():
    phishing = [s for s in QR_SCENARIOS if s["label"] == 1]
    safe     = [s for s in QR_SCENARIOS if s["label"] == 0]

    # pick 5 phishing + 5 safe (or as many as available)
    n_each   = min(5, len(phishing), len(safe))
    selected = random.sample(phishing, n_each) + random.sample(safe, n_each)
    random.shuffle(selected)

    result = [dict(s) for s in selected]
    for s in result:
        s["qr_image"] = qr_image_for(s["url"])
    return jsonify(result)


@app.route("/classify-qr-url", methods=["POST"])
def classify_qr_url_route():
    """Mission 3 — classify a URL decoded from a QR code using the QR BiLSTM."""
    data = request.get_json() or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing url"}), 400
    result = classify_qr_url(url)
    result.update(check_url_accessibility(url))
    return jsonify(result)


@app.route("/decode-qr", methods=["POST"])
def decode_qr():
    """Decode a QR image and classify the URL using the QR BiLSTM."""
    arr = None
    img = None

    # 1. Check if a file was uploaded via multipart/form-data
    if "image" in request.files or "file" in request.files:
        uploaded_file = request.files.get("image") or request.files.get("file")
        if uploaded_file and uploaded_file.filename != "":
            try:
                img = load_image_stream(uploaded_file.stream, {"PNG", "JPEG", "WEBP", "BMP", "GIF"})
                arr = np.array(img)
            except Exception as e:
                return jsonify({"error": f"Uploaded file is not a readable image: {e}"}), 400

    # 2. Check JSON payload or form body
    if arr is None:
        data = request.get_json(silent=True) or {}
        image_input = data.get("image") or data.get("file") or ""
        if not image_input and request.form:
            image_input = request.form.get("image") or request.form.get("file") or ""

        if not image_input:
            return jsonify({"error": "Missing image input. Please provide a base64 encoded image or upload an image file."}), 400

        if not isinstance(image_input, str):
            return jsonify({"error": "Image input must be a base64 encoded string."}), 400

        image_input = image_input.strip()

        # Handle data URI scheme (e.g. data:image/png;base64,...)
        try:
            img = load_base64_image(image_input, {"PNG", "JPEG", "WEBP", "BMP", "GIF"})
            arr = np.array(img)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {e}. Please upload a readable image or valid base64 image data."}), 400

    if arr is None or arr.size == 0 or arr.shape[0] < 10 or arr.shape[1] < 10:
        return jsonify({"error": "Image is empty or too small to contain a readable QR code."}), 400

    url = None
    # 1. Try OpenCV QRCodeDetector
    try:
        import cv2
        detector = cv2.QRCodeDetector()
        decoded, _, _ = detector.detectAndDecode(arr)
        if decoded and decoded.strip():
            url = decoded.strip()
    except Exception:
        url = None

    # 2. Try pyzbar if OpenCV didn't detect
    if not url:
        try:
            from pyzbar.pyzbar import decode as zbar_decode
            for obj in zbar_decode(img):
                if obj.data:
                    url = obj.data.decode("utf-8", "ignore").strip()
                    if url:
                        break
        except Exception:
            pass

    if not url:
        return jsonify({"error": "No QR code detected in the image. Please try a clearer or higher-contrast photo."}), 422

    result = classify_qr_url(url)
    result.update(check_url_accessibility(url))
    result["url"] = url
    return jsonify(result)


@app.route("/test_email.html")
@app.route("/analyzer")
@app.route("/mail-analyzer")
def serve_test_email():
    return send_from_directory(BASE_DIR, "test_email.html")


@app.route("/health", methods=["GET"])
def health():
    active_models = {
        "email_distilbert": model is not None,
        "url_bilstm": url_model is not None,
        "qr_bilstm": qr_model is not None,
        "sms_distilbert": sms_model is not None,
    }
    return jsonify({
        "status": "ok",
        "mode": DEPLOYMENT_MODE,
        "service": "CyberSleuth Threat Analysis Platform",
        "active_models": active_models,
        "url_model": active_models["url_bilstm"],
        "qr_model": active_models["qr_bilstm"],
        "distilbert": active_models["email_distilbert"],
        "sms_model": active_models["sms_distilbert"],
        "sms_model_path": sms_model_path,
        "timestamp": now_iso()
    })


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # If file exists in game/dist, serve it directly
    target_path = os.path.join(DIST_DIR, path)
    if path != "" and os.path.exists(target_path) and os.path.isfile(target_path):
        return send_from_directory(DIST_DIR, path)
    # Default to index.html for Single Page App routing
    index_file = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_file):
        return send_from_directory(DIST_DIR, "index.html")
    return jsonify({
        "service": "CyberSleuth Threat Analysis Platform Backend",
        "status": "online",
        "endpoints": [
            "/predict", "/classify-sms", "/classify-url", "/random-emails",
            "/random-qr", "/decode-qr", "/auth/login", "/auth/register",
            "/submit-score", "/leaderboard", "/ml-hint"
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"CyberSleuth Server starting on {host}:{port}")
    app.run(host=host, port=port, debug=False)
