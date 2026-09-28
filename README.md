# Kaggle SDXL GPU backend

An ephemeral **text-to-image** backend for a Telegram bot, designed to run on a
**free Kaggle GPU** session. It loads any single-file SDXL `.safetensors`
checkpoint (supplied at runtime, not bundled), serves a small FastAPI job API
plus an optional Gradio web app, exposes them over a Cloudflare Quick Tunnel,
and registers that public URL with a companion Cloudflare Worker that fronts the
Telegram bot.

```
Telegram ──▶ Cloudflare Worker ──▶ (this) Kaggle tunnel ──▶ SDXL on GPU ──▶ image ──▶ Telegram
```

## What's here

- `kaggle_backend.ipynb` — the notebook you run on Kaggle (Run All).
- `backend/app.py` — the FastAPI app (job queue, `/generate`, `/status`,
  `/result`, `/health`, and a mountable Gradio UI at `/app`). Also pushes the
  finished image to Telegram and self-deletes it after 30s.

## Model is NOT included

The checkpoint is never committed to this repo. You provide it at runtime via a
Kaggle **secret** named `MODEL_URL` — a direct download link to your SDXL
`.safetensors` file.

## Setup (in the Kaggle notebook)

1. **Settings → Accelerator → GPU** (T4 x2 or P100).
2. **Settings → Internet → On**.
3. **Add-ons → Secrets** — add and enable:
   - `WORKER_REGISTER_URL` = `https://<your-worker>.workers.dev/register-backend`
   - `REGISTRATION_SECRET` = the same secret configured in Cloudflare
   - `TELEGRAM_BOT_TOKEN` = your Telegram bot token
   - `MODEL_URL` = direct link to your `.safetensors` checkpoint
4. **Run All**. When it prints `Registered temporary backend: https://...`,
   open the bot and send `/generate <prompt>`.

A Kaggle session stays alive up to ~9 hours. The keep-alive cell holds the
kernel busy so the tunnel stays registered; interrupt it to shut down.

## Security notes

- Secrets are read from Kaggle Secrets at runtime; nothing sensitive is stored
  in this public repo.
- The Cloudflare tunnel URL is public but unguessable and short-lived; the
  Worker gates who may trigger generation via `ALLOWED_TELEGRAM_USER_IDS`.
