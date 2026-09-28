import asyncio
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/content/sdxl_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RETENTION_SECONDS = int(os.getenv("RETENTION_SECONDS", "3600"))
# Optional Telegram delivery: when TELEGRAM_BOT_TOKEN is set and a job carries a
# chat_id, the backend uploads the finished PNG directly via sendPhoto. This
# closes the loop without the Worker polling or holding a request open.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
# Device the pipeline currently runs on ("cuda" or "cpu"). Set by set_pipeline
# and switchable at runtime via /device so you can fall back to CPU (slow but
# free, burns no GPU quota) without reloading the model.
DEVICE = "cpu"
JOBS: dict[str, dict[str, Any]] = {}
QUEUE: asyncio.Queue = asyncio.Queue(maxsize=8)
PIPELINE = None
WORKER_TASK = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str = Field(default="", max_length=1000)
    width: int = Field(default=1024, ge=512, le=1024, multiple_of=64)
    height: int = Field(default=1024, ge=512, le=1024, multiple_of=64)
    steps: int = Field(default=25, ge=10, le=40)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=15.0)
    ip_adapter_scale: float = Field(default=0.6, ge=0.0, le=1.0)
    seed: int = Field(default=-1, ge=-1, le=2147483647)
    # Telegram chat to deliver the finished image to. When set, the backend
    # pushes the PNG directly with sendPhoto once generation completes, so the
    # Worker never has to poll or hold a request open.
    chat_id: int | None = Field(default=None)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value.strip()


app = FastAPI(title="SDXL IP-Adapter ephemeral backend")


def set_pipeline(pipeline, device: str | None = None) -> None:
    global PIPELINE, DEVICE
    PIPELINE = pipeline
    if device:
        DEVICE = device
    else:
        # Infer from the model's current device.
        try:
            DEVICE = "cuda" if next(pipeline.unet.parameters()).is_cuda else "cpu"
        except Exception:
            DEVICE = "cpu"


def move_pipeline(device: str) -> str:
    """Move the loaded pipeline to 'cuda' or 'cpu' at runtime. Returns the
    device actually in effect. Used by /device so you can drop to CPU (free, no
    GPU quota) or back to GPU without reloading weights."""
    global DEVICE
    if PIPELINE is None:
        raise RuntimeError("model pipeline is not loaded")
    device = device.lower().strip()
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this runtime")
    PIPELINE.to(device)
    DEVICE = device
    return DEVICE


def cleanup() -> None:
    cutoff = time.time() - RETENTION_SECONDS
    for job_id, job in list(JOBS.items()):
        if job.get("updated_at", 0) < cutoff:
            if job.get("path"):
                Path(job["path"]).unlink(missing_ok=True)
            JOBS.pop(job_id, None)


def run_generation(job_id: str, payload: GenerateRequest) -> Path:
    if PIPELINE is None:
        raise RuntimeError("model pipeline is not loaded")
    import torch
    gen_device = "cuda" if DEVICE == "cuda" else "cpu"
    generator = None if payload.seed == -1 else torch.Generator(device=gen_device).manual_seed(payload.seed)
    # set_ip_adapter_scale only applies when an IP-Adapter is actually loaded.
    # For pure text-to-image the adapter is off, so guard the call.
    try:
        PIPELINE.set_ip_adapter_scale(payload.ip_adapter_scale)
    except (ValueError, AttributeError):
        pass
    result = PIPELINE(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        num_inference_steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        generator=generator,
    )
    path = OUTPUT_DIR / f"{job_id}.png"
    result.images[0].save(path)
    return path


def deliver_to_telegram(chat_id: int, path: Path, caption: str) -> None:
    """Upload the finished PNG via sendPhoto, post an English warning that the
    image will be removed, then delete BOTH the photo and the warning after 30
    seconds. Best-effort: failures are swallowed so a delivery hiccup never
    crashes the GPU worker (the image is still retrievable via /result)."""
    if not TELEGRAM_BOT_TOKEN:
        return
    import requests

    photo_message_id = None
    try:
        with open(path, "rb") as image:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"photo": (path.name, image, "image/png")},
                timeout=60,
            )
        photo_message_id = resp.json().get("result", {}).get("message_id")
    except Exception:
        pass

    # English warning that both the image and this notice self-destruct in 30s.
    notice_message_id = None
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": "\U0001F3A8 New image generated.\n\u26A0\uFE0F This image will be deleted in 30 seconds.",
            },
            timeout=30,
        )
        notice_message_id = resp.json().get("result", {}).get("message_id")
    except Exception:
        pass

    # Delete both the photo and the warning after 30 seconds.
    to_delete = [mid for mid in (photo_message_id, notice_message_id) if mid is not None]
    if to_delete:
        def _delete_later(message_ids: list[int]) -> None:
            time.sleep(30)
            for mid in message_ids:
                try:
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage",
                        data={"chat_id": chat_id, "message_id": mid},
                        timeout=30,
                    )
                except Exception:
                    pass
        threading.Thread(target=_delete_later, args=(to_delete,), daemon=True).start()


def notify_failure(chat_id: int, error: str) -> None:
    """Tell the user their job failed, so a crash never looks like silence."""
    if not TELEGRAM_BOT_TOKEN:
        return
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": chat_id, "text": f"Generation failed: {error}"},
            timeout=30,
        )
    except Exception:
        pass


async def gpu_worker() -> None:
    while True:
        job_id, payload = await QUEUE.get()
        JOBS[job_id].update(status="generating", updated_at=time.time())
        try:
            path = await asyncio.to_thread(run_generation, job_id, payload)
            JOBS[job_id].update(status="completed", path=str(path), updated_at=time.time())
            if payload.chat_id is not None:
                await asyncio.to_thread(
                    deliver_to_telegram,
                    payload.chat_id,
                    path,
                    payload.prompt,
                )
        except Exception as exc:
            JOBS[job_id].update(status="failed", error=str(exc)[:300], updated_at=time.time())
            if payload.chat_id is not None and TELEGRAM_BOT_TOKEN:
                await asyncio.to_thread(notify_failure, payload.chat_id, str(exc)[:300])
        finally:
            QUEUE.task_done()
            cleanup()


@app.on_event("startup")
async def startup() -> None:
    global WORKER_TASK
    WORKER_TASK = asyncio.create_task(gpu_worker())


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": PIPELINE is not None, "queue_size": QUEUE.qsize(), "device": DEVICE}


class DeviceRequest(BaseModel):
    device: str = Field(pattern="^(cuda|cpu)$")


@app.post("/device")
def set_device(payload: DeviceRequest):
    """Switch the pipeline between GPU and CPU at runtime. CPU is slow (minutes
    per image) but free and burns no GPU quota."""
    if PIPELINE is None:
        raise HTTPException(503, "model pipeline is not loaded")
    try:
        effective = move_pipeline(payload.device)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True, "device": effective}


@app.post("/shutdown")
def shutdown():
    """Terminate this backend process, which ends the hosting Kaggle/Colab
    session and stops consuming any quota. Best-effort: we clear the KV
    registration is the Worker's job; here we just exit hard after replying.
    A background timer calls os._exit so the HTTP response is flushed first."""
    def _die():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"ok": True, "message": "backend shutting down"}


@app.post("/generate", status_code=202)
async def generate(payload: GenerateRequest):
    if PIPELINE is None:
        raise HTTPException(503, "model pipeline is not loaded")
    job_id = uuid.uuid4().hex
    now = time.time()
    JOBS[job_id] = {"status": "queued", "created_at": now, "updated_at": now}
    try:
        QUEUE.put_nowait((job_id, payload))
    except asyncio.QueueFull:
        JOBS.pop(job_id, None)
        raise HTTPException(429, "generation queue is full")
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {key: value for key, value in job.items() if key != "path"}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"job is {job['status']}")
    return FileResponse(job["path"], media_type="image/png", filename=f"{job_id}.png")


def attach_gradio(mount_path: str = "/app"):
    """Mount a Gradio UI on the FastAPI app at `mount_path`, driven by the same
    in-process pipeline. Kept as a function (not import-time) so CI and tests
    never need Gradio installed — the Colab notebook calls this after the model
    is loaded. The UI is served over the same Cloudflare tunnel, so the Worker
    can expose it as a Telegram Mini App (web_app) button pointing to
    `<tunnel>/app`."""
    import gradio as gr

    max_seed = 2147483647

    def _generate(prompt, negative, steps, cfg, width, height, ip_scale, seed, randomize):
        if PIPELINE is None:
            raise gr.Error("Model pipeline is not loaded yet.")
        import torch
        if randomize or seed is None or int(seed) < 0:
            seed = int(torch.randint(0, max_seed, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        try:
            PIPELINE.set_ip_adapter_scale(float(ip_scale))
        except Exception:
            pass
        image = PIPELINE(
            prompt=prompt,
            negative_prompt=negative,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator,
        ).images[0]
        return image, seed

    with gr.Blocks(title="Anime Desire Illustrious") as blocks:
        gr.Markdown("## Anime Desire Illustrious (SDXL) — Telegram Mini App")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=3,
                    value="1girl, solo, masterpiece, best quality, highly detailed")
                negative = gr.Textbox(label="Negative", lines=2,
                    value="lowres, worst quality, low quality, bad anatomy, bad hands, text, watermark")
                with gr.Row():
                    steps = gr.Slider(10, 40, value=25, step=1, label="Steps")
                    cfg = gr.Slider(1, 15, value=7, step=0.5, label="CFG")
                with gr.Row():
                    width = gr.Slider(512, 1024, value=1024, step=64, label="Width")
                    height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
                ip_scale = gr.Slider(0, 1, value=0.6, step=0.05, label="IP-Adapter scale")
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
                    randomize = gr.Checkbox(value=True, label="Randomize")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Output", type="pil")
                out_seed = gr.Number(label="Used seed", precision=0)
        btn.click(_generate,
            inputs=[prompt, negative, steps, cfg, width, height, ip_scale, seed, randomize],
            outputs=[out_img, out_seed])

    return gr.mount_gradio_app(app, blocks, path=mount_path)
