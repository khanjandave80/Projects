# CyberSleuth Academy

CyberSleuth Academy is a browser-based cybersecurity awareness game built for a Semester 7 major project. Players investigate realistic social-engineering scenarios and learn to recognize phishing emails, malicious URLs, QR-code scams, and fraudulent text messages.

The application combines a Flask API, a Vite frontend, locally loaded machine-learning models, and rule-based checks. It is an educational demonstration, not a replacement for security products or professional incident response.

## Features

Five missions cover common attack vectors:

1. **Email phishing** — Review email scenarios and inspect the classifier's verdict, confidence, and signals.
2. **Malicious URLs** — Classify URLs as Safe, Phishing, or Suspicious with a character-level BiLSTM.
3. **QR quishing** — Decode a QR code and assess its destination with the QR URL classifier.
4. **SMS smishing** — Review suspicious messages using rule-based checks.
5. **Mail analyzer** — Paste email text or provide a screenshot for OCR, field extraction, and threat analysis.

Players can create accounts, earn scores, and compare results on the leaderboard. Performance categories range from Normal to Elite.

## Technology

- **Frontend:** Vanilla JavaScript and Vite
- **Backend:** Python, Flask, Flask-CORS, and SQLite
- **Email analysis:** Fine-tuned DistilBERT when its weights are installed; local heuristics otherwise
- **URL and QR analysis:** PyTorch character-level BiLSTM models
- **SMS analysis:** Rule-based heuristics
- **Screenshot OCR:** Windows Media OCR through PowerShell, with Tesseract as a fallback
- **Deployment:** Docker/Render or a temporary Cloudflare Quick Tunnel

## Repository contents

```text
app.py                    Flask API and inference
game/                     Frontend source and Vite configuration
models/url_bilstm/        URL model weights and configuration
models/qr_BiLSTM/         QR model weights and configuration
models/Email_DistilBERT/  Tokenizer and model configuration
dataset/game_emails.json  Curated email scenarios for the game
train_*.py                Model training scripts
prepare_*.py              Dataset preparation scripts
evaluate*.py              Evaluation scripts
Dockerfile                Container build
render.yaml               Render service configuration
ocr_engine.ps1            Windows OCR helper
start_backend.bat         Local Windows launcher
start_online_tunnel.bat   Cloudflare tunnel launcher
```

Large email-model weights and local research data are not included. Raw datasets, reports, generated training outputs, virtual environments, databases, and secrets should not be committed unless their licenses and contents have been reviewed. The repository's `.gitignore` and `.dockerignore` exclude these local files.

## Requirements

- Python 3.12 recommended
- Node.js 18 or newer with npm
- Windows PowerShell for Windows Media OCR
- Tesseract installed and available on `PATH` if using the OCR fallback
- `cloudflared` on `PATH` for the optional tunnel

## Run locally

From the project root, create and activate a Python environment, then install the backend dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Build the frontend:

```powershell
Set-Location game
npm ci
npm run build
Set-Location ..
```

Start the backend:

```powershell
python app.py
```

Open [http://localhost:5000](http://localhost:5000). The Windows launcher `start_backend.bat` performs environment detection and starts the backend.

If no admin passwords are configured in local mode, the application generates temporary passwords on startup and prints them in the backend console. They are rotated on the next startup. To keep credentials stable, set `CS_ADMIN_1_PASSWORD` and `CS_ADMIN_2_PASSWORD` to strong values before starting the app.

## Machine-learning weights

The email classifier's `model.safetensors` file is larger than GitHub's regular file limit and is intentionally excluded. Obtain the trained weights from the project's approved model host and place them in `models/Email_DistilBERT/` alongside the included configuration and tokenizer files. **A public model download link has not yet been added.**

The URL and QR BiLSTM checkpoints are small and are kept in the repository. The `/health` endpoint reports which models loaded. Without the email weights, email analysis uses local heuristics instead of DistilBERT.

## Share a demo with a Cloudflare Quick Tunnel

On Windows, run `start_online_tunnel.bat`. It builds the frontend, starts the Flask backend, checks `/health`, and starts a temporary Cloudflare tunnel. Keep both opened command windows running while the demo is shared; the tunnel URL changes on each run.

The tunnel launcher generates admin passwords if they are not supplied as environment variables and prints them in its console. Keep these credentials private. The launcher waits for the email, URL, and QR models to load; install the email weights before using this flow.

## Deploy with Docker and Render

The included `Dockerfile` builds the frontend and runs Flask on port `7860`. `render.yaml` configures the Docker service and `/health` health check.

For an online deployment, configure both environment variables in the hosting dashboard:

| Variable | Requirement |
| --- | --- |
| `CS_ADMIN_1_PASSWORD` | Strong password, at least 16 characters |
| `CS_ADMIN_2_PASSWORD` | Strong password, at least 16 characters |

Online mode refuses to start if either password is missing or shorter than 16 characters. Do not put real passwords in source files or commit them to GitHub.

Other supported settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PORT` | Flask listening port | `5000` locally; `7860` in Docker |
| `HOST` | Bind address | `0.0.0.0` |
| `DEPLOYMENT_MODE` | `online` enforces explicit admin passwords | `local` when unset |

## Security and privacy

- The SQLite database is created locally under `data/` and must not be published; it stores account and score data.
- Do not upload `.env` files, passwords, raw email corpora, reports, or generated model outputs without reviewing their contents and licenses.
- The online mode requires unique, strong administrator passwords. A public tunnel makes the running application reachable over the internet; use it only for a controlled demo.
- Treat classifier results as educational guidance. A “Safe” verdict is not a guarantee that a message or URL is safe.

## Academic project

Built as a Semester 7 major project exploring applied machine learning, cybersecurity awareness, and gamified learning.
