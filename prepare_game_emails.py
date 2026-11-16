import csv, json, re, random, os

csv.field_size_limit(10**7)

def clean_text(text):
    text = re.sub(r'=\r?\n', '', text)           # quoted-printable line breaks
    text = re.sub(r'=[0-9A-Fa-f]{2}', '', text)  # quoted-printable encoded chars
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML tags
    text = re.sub(r'[ \t]+', ' ', text)           # collapse spaces
    text = re.sub(r'\n{3,}', '\n\n', text)        # collapse blank lines
    return text.strip()

def extract_subject(text):
    m = re.search(r'Subject[:\s]+(.+?)(\n|$)', text, re.IGNORECASE)
    if m:
        s = m.group(1).strip()[:120]
        if len(s) > 5:
            return s
    # fallback: first clean line with spaces (real sentence)
    for line in text.split('\n'):
        line = line.strip()
        if len(line) > 15 and ' ' in line and not re.search(r'^[a-z]{10,}$', line):
            return line[:120]
    return 'No Subject'

def extract_sender(text):
    m = re.search(r'From[:\s]+(.+?)(\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:100]
    return ''

def extract_url(text):
    m = re.search(r'https?://[^\s<>"\']+', text)
    if m:
        return m.group(0)[:200]
    return ''

def extract_body(text, subject, sender):
    body = re.sub(r'^(Subject|From|To|Date|Reply-To|Return-Path|Received|MIME|Content)[:\s].+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    return body if len(body) >= 100 else None

def is_readable(text):
    # reject if too many consecutive chars without spaces (garbled)
    words = text.split()
    if len(words) < 10:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 12:
        return False
    # reject if mostly lowercase run-on (no spaces)
    if re.search(r'[a-z]{20,}', text):
        return False
    return True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT    = os.path.join(BASE_DIR, 'data', 'fraud_email_.csv')
OUTPUT   = os.path.join(BASE_DIR, 'data', 'game_emails.json')

phishing_pool = []
safe_pool     = []

with open(INPUT, encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    next(reader)  # skip header: Text, Class
    for row in reader:
        if len(row) < 2:
            continue
        text  = clean_text(row[0])
        label = row[1].strip()
        if label not in ('0', '1'):
            continue
        if len(text) < 80:
            continue

        subject = extract_subject(text)
        sender  = extract_sender(text)
        url     = extract_url(text)
        body    = extract_body(text, subject, sender)

        if not body:
            continue

        # Skip garbled / unreadable emails
        if re.search(r'escapenumber|=3D|=20|\\x[0-9a-f]{2}|&nbsp;|&amp;|&#', body, re.IGNORECASE):
            continue
        if not is_readable(body):
            continue
        # Subject must look like a real sentence
        if subject == 'No Subject' or len(subject) < 8 or not re.search(r'[a-zA-Z]{3,}\s[a-zA-Z]{2,}', subject):
            continue
        # Body must have proper sentence structure
        if not re.search(r'[A-Z][a-z].*[.!?]', body):
            continue

        email = {
            'subject': subject,
            'sender' : sender,
            'urls'   : url,
            'body'   : body[:1200],
            'label'  : int(label)
        }

        if label == '1' and len(phishing_pool) < 500:
            phishing_pool.append(email)
        elif label == '0' and len(safe_pool) < 500:
            safe_pool.append(email)

        if len(phishing_pool) >= 500 and len(safe_pool) >= 500:
            break

print(f'Collected: {len(phishing_pool)} phishing, {len(safe_pool)} safe')

# Pick 250 from each
random.seed(42)
selected = random.sample(phishing_pool, min(250, len(phishing_pool))) + \
           random.sample(safe_pool,     min(250, len(safe_pool)))

random.shuffle(selected)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

print(f'Saved {len(selected)} emails to {OUTPUT}')
print(f'Phishing: {sum(1 for e in selected if e["label"]==1)}')
print(f'Safe    : {sum(1 for e in selected if e["label"]==0)}')
