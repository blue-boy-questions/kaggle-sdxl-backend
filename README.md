# Kaggle SDXL GPU backend + Telegram Worker

<p>
  <a href="https://deploy.workers.cloudflare.com/?url=https://github.com/blue-boy-questions/kaggle-sdxl-backend/tree/main/worker">
    <img src="https://deploy.workers.cloudflare.com/button" alt="Deploy to Cloudflare Workers">
  </a>
  <a href="https://www.kaggle.com/kernels/welcome?src=https://github.com/blue-boy-questions/kaggle-sdxl-backend/blob/main/kaggle_backend.ipynb">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle">
  </a>
</p>

A Telegram text-to-image bot backed by a **free Kaggle GPU** session. A Cloudflare
Worker fronts the bot; the Kaggle notebook loads any single-file SDXL
`.safetensors` checkpoint (supplied at runtime, **not** bundled), serves a small
FastAPI job API plus an optional Gradio web app, exposes them over a Cloudflare
Quick Tunnel, and registers that public URL with the Worker.

```text
Telegram ──▶ Cloudflare Worker ──▶ Cloudflare Quick Tunnel ──▶ Kaggle FastAPI ──▶ SDXL on GPU ──▶ image ──▶ Telegram
```

## What's here

- `worker/` — the Cloudflare Worker (Telegram webhook, allow-list, secure backend
  registration, health probing, Mini App button). Deploy this first.
- `kaggle_backend.ipynb` — the notebook you Run All on Kaggle.
- `backend/app.py` — the FastAPI app (`/generate`, `/status`, `/result`,
  `/health`, and a mountable Gradio UI at `/app`). Pushes the finished image to
  Telegram and self-deletes it after 30s.
- `tests/` — lightweight tests that never download model weights.

## Order of setup: Worker first, then Kaggle

The Kaggle notebook's last step **registers** its tunnel URL with the Worker, so
the Worker must already be deployed.

### 1) Deploy the Cloudflare Worker

1. Click **Deploy to Cloudflare Workers** above. Cloudflare reads
   `worker/wrangler.toml` and provisions the `BACKEND_KV` namespace automatically.
2. Keep the build command empty; deploy command is `npm run deploy`.
3. After the first deploy, add these secrets (dashboard → Worker → Settings →
   Variables, or via CLI):

   ```bash
   wrangler secret put TELEGRAM_BOT_TOKEN
   wrangler secret put TELEGRAM_WEBHOOK_SECRET   # a random string you choose
   wrangler secret put REGISTRATION_SECRET       # a random string; reused in Kaggle
   ```

4. Optional `[vars]`: `ALLOWED_TELEGRAM_USER_IDS` (comma-separated numeric IDs)
   and `COLAB_NOTEBOOK_URL` (a notebook link for the cold-start button).
5. Set the Telegram webhook to `https://<worker>/telegram/webhook`, passing the
   same `TELEGRAM_WEBHOOK_SECRET` as Telegram's `secret_token`:

   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/setWebhook" \
     -d "url=https://<worker>.workers.dev/telegram/webhook" \
     -d "secret_token=<TELEGRAM_WEBHOOK_SECRET>"
   ```

### 2) Run the Kaggle GPU backend

1. Click **Open in Kaggle** above (or import the notebook from this repo).
2. **Settings → Accelerator → GPU** (T4 x2 or P100) and **Settings → Internet → On**.
3. **Add-ons → Secrets** — add and enable:
   - `WORKER_REGISTER_URL` = `https://<your-worker>.workers.dev/register-backend`
   - `REGISTRATION_SECRET` = the same secret set in the Worker
   - `TELEGRAM_BOT_TOKEN` = your bot token
   - `MODEL_URL` = direct download link to your SDXL `.safetensors` checkpoint
4. **Run All**. When it prints `Registered temporary backend: https://...`, open
   the bot and send `/generate <prompt>`.

A Kaggle session stays alive up to ~9 hours; the keep-alive cell holds the kernel
busy so the tunnel stays registered. Interrupt it to shut down.

## Bot commands

- `/start`, `/help`, `/status`, `/generate <prompt>`

## Model is NOT in this repo

The checkpoint is never committed. You supply it at runtime via the Kaggle secret
`MODEL_URL`. `.gitignore` blocks `*.safetensors` and friends.

## Security notes

- Secrets live in Kaggle Secrets / Cloudflare secrets, never in this public repo.
- The tunnel URL is public but unguessable and short-lived; the Worker gates who
  may trigger generation via `ALLOWED_TELEGRAM_USER_IDS`.

## Local verification

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install fastapi httpx pytest requests
pytest -q
node --check worker/src/index.js
node --test tests/worker.test.mjs
python3 -m json.tool kaggle_backend.ipynb >/dev/null
```
