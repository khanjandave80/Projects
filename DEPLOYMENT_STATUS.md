# CyberSleuth Deployment Status

## Completed

- [x] Flask binds to `0.0.0.0`, honors `PORT`, and exposes `/health` with the loaded model states.
- [x] The Vite frontend is built and served by Flask on the same origin as its APIs.
- [x] Production-hosted frontend requests use `window.location.origin`; Vite development uses the local Flask backend.
- [x] The Mail Analyzer accepts PNG/JPG/JPEG/WEBP uploads, drag-and-drop, clipboard images, and pasted email text.
- [x] Windows Media OCR is used locally on Windows, with local Tesseract fallback when available; OCR telemetry is returned with the classification.
- [x] QR scenarios and image generation use local `/random-qr` and `/generate-qr` endpoints; `/decode-qr` validates uploaded and base64 images.
- [x] `start_online_tunnel.bat` rebuilds the frontend, selects an installed compatible Python environment (preferring CUDA), waits for `/health`, then opens a Cloudflare Quick Tunnel.
- [x] Online mode requires strong admin passwords; the Windows tunnel launcher generates temporary random passwords unless they are supplied through environment variables.
- [x] `start_backend.bat` selects an installed compatible Python environment (preferring CUDA) without a fixed Python installation path.
- [x] The analyzer prompts before analysis and dashboard exit, and warns before leaving with unsaved input.

## Run locally or share temporarily

1. Install the project Python dependencies, Node.js/npm, and `cloudflared`.
2. Ensure port 5000 is free, then run `start_online_tunnel.bat` from the project directory. The script refuses to tunnel through an already-running server; set `PORT` to a free port if needed.
3. Keep the generated administrator passwords private; the launcher displays them locally. To retain credentials between runs, set `CS_ADMIN_1_PASSWORD` and `CS_ADMIN_2_PASSWORD` to unique values of at least 16 characters before starting the script.
4. The tunnel starts only after `/health` confirms the email, URL, and QR models are loaded.
5. Keep both the backend window and tunnel window open. The `https://*.trycloudflare.com` URL printed by `cloudflared` is temporary and changes when a new Quick Tunnel is started.
6. Open the printed URL on the computer or mobile device. Verify `/health` at the same host if needed.

The Cloudflare URL is not a permanent deployment: the host computer, backend, and tunnel must remain online. A durable URL requires a named Cloudflare Tunnel or a hosted deployment.

An already-running Python process does not load source changes automatically. The launcher refuses to reuse an occupied port, so close the prior backend/tunnel windows yourself before starting the updated version.

## Verification

- `game`: `npm run build` passes (Vite reports the existing advisory that the JS bundle exceeds 500 kB).
- `python -m py_compile app.py` passes.
- Local smoke checks passed for OCR telemetry extraction/classification, malformed QR input validation, and QR generation/decoding through the local endpoints.
- A live public URL must be obtained from the running `cloudflared` process; it cannot be predetermined or guaranteed after that process stops.
