const BACKEND_KEY = "active_backend";
const MAX_PROMPT_LENGTH = 1000;
const DEFAULT_BACKEND_TTL = 7200;
const PENDING_TTL = 300; // seconds a "waiting for your prompt" state lives

export function allowedUser(env, userId) {
  const configured = (env.ALLOWED_TELEGRAM_USER_IDS || "").trim();
  if (!configured) return true;
  return configured.split(",").map((value) => value.trim()).includes(String(userId));
}

export function parseCommand(text = "") {
  const [raw = "", ...rest] = text.trim().split(/\s+/);
  return {
    command: raw.split("@")[0].toLowerCase(),
    argument: rest.join(" "),
  };
}

export function validateGenerate(input) {
  if (!input || typeof input.prompt !== "string" || !input.prompt.trim()) {
    throw new Error("prompt is required");
  }
  if (input.prompt.length > MAX_PROMPT_LENGTH) {
    throw new Error("prompt is too long");
  }

  const bounded = (value, fallback, minimum, maximum) => {
    const number = Number(value ?? fallback);
    if (!Number.isFinite(number) || number < minimum || number > maximum) {
      throw new Error(`value must be between ${minimum} and ${maximum}`);
    }
    return number;
  };

  return {
    prompt: input.prompt.trim(),
    negative_prompt: String(input.negative_prompt || "").slice(0, MAX_PROMPT_LENGTH),
    width: bounded(input.width, 1024, 512, 1024),
    height: bounded(input.height, 1024, 512, 1024),
    steps: bounded(input.steps, 25, 10, 40),
    guidance_scale: bounded(input.guidance_scale, 7, 1, 15),
    ip_adapter_scale: bounded(input.ip_adapter_scale, 0.6, 0, 1),
    seed: bounded(input.seed, -1, -1, 2147483647),
  };
}

async function telegram(env, method, body) {
  return fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function reply(env, chatId, text, extra = {}) {
  return telegram(env, "sendMessage", { chat_id: chatId, text, ...extra });
}

function answerCallback(env, callbackId, text = "") {
  return telegram(env, "answerCallbackQuery", { callback_query_id: callbackId, text });
}

// Glass (inline) main menu. Buttons carry callback_data the Worker handles;
// the Web App button only renders when a live backend/tunnel exists.
export function mainMenu(env, backend) {
  const rows = [
    [{ text: "🎨 Generate", callback_data: "gen" }],
    [{ text: "📊 Status", callback_data: "status" }],
  ];
  if (backend?.url) {
    const dev = backend.device === "cuda" ? "cpu" : "cuda";
    const label = dev === "cpu" ? "🐢 Switch to CPU (free)" : "⚡ Switch to GPU";
    rows.push([{ text: label, callback_data: `dev:${dev}` }]);
    rows.push([{ text: "🖼 Open Web App", web_app: { url: `${backend.url}/app` } }]);
    rows.push([{ text: "🛑 Stop session", callback_data: "stop" }]);
  } else if (kaggleConfigured(env)) {
    // No backend live, but we can remotely start one via the Kaggle API.
    rows.push([{ text: "🐢 Wake (CPU, free)", callback_data: "wake:cpu" }]);
    rows.push([{ text: "⚡ Wake (GPU, fast)", callback_data: "wake:gpu" }]);
  } else {
    const colab = (env.COLAB_NOTEBOOK_URL || "").trim();
    if (colab) rows.push([{ text: "▶️ Start backend (Kaggle/Colab)", url: colab }]);
  }
  return { reply_markup: { inline_keyboard: rows } };
}

function kaggleConfigured(env) {
  return Boolean(env.KAGGLE_USERNAME && env.KAGGLE_KEY && env.KAGGLE_KERNEL);
}

// --- Kaggle API helpers (kernels pull/push to remotely start a batch run) ---
function kaggleAuth(env) {
  return "Basic " + btoa(`${env.KAGGLE_USERNAME}:${env.KAGGLE_KEY}`);
}

async function kaggleGet(env, path) {
  const r = await fetch(`https://www.kaggle.com/api/v1${path}`, {
    headers: { authorization: kaggleAuth(env), accept: "application/json" },
    signal: AbortSignal.timeout(20000),
  });
  return r.json();
}

// Push the kernel (Run All / batch) with the requested accelerator. Returns the
// parsed API response. device: "cpu" | "gpu".
async function kaggleWake(env, device) {
  // Pull current source + metadata so we re-push faithfully, only flipping GPU.
  const pull = await kaggleGet(env, `/kernels/pull/${env.KAGGLE_KERNEL}`);
  const meta = typeof pull.metadata === "string" ? JSON.parse(pull.metadata) : pull.metadata;
  const source = pull.blob?.source ?? "";

  const [owner, slug] = env.KAGGLE_KERNEL.split("/");
  const body = {
    id: meta.id,
    slug: env.KAGGLE_KERNEL,
    newTitle: meta.title || slug,
    text: source,
    language: "python",
    kernelType: meta.kernelType || "notebook",
    isPrivate: meta.isPrivate ?? false,
    enableGpu: device === "gpu",
    enableTpu: false,
    enableInternet: true, // required to download the model + open the tunnel
    datasetDataSources: meta.datasetDataSources || [],
    competitionDataSources: meta.competitionDataSources || [],
    kernelDataSources: meta.kernelDataSources || [],
    modelDataSources: meta.modelDataSources || [],
    categoryIds: meta.categoryIds || [],
  };

  const r = await fetch("https://www.kaggle.com/api/v1/kernels/push", {
    method: "POST",
    headers: {
      authorization: kaggleAuth(env),
      accept: "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30000),
  });
  return { status: r.status, body: await r.json() };
}


async function activeBackend(env) {
  const record = await env.BACKEND_KV.get(BACKEND_KEY, "json");
  if (!record || record.expires_at <= Date.now()) return null;

  try {
    const response = await fetch(`${record.url}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return response.ok ? record : null;
  } catch {
    return null;
  }
}

// Submit a validated prompt to the live backend. Returns a user-facing message.
async function submitGeneration(env, chatId, promptText) {
  let payload;
  try {
    payload = validateGenerate({ prompt: promptText });
  } catch (error) {
    return `Invalid request: ${error.message}`;
  }

  const backend = await activeBackend(env);
  if (!backend) {
    return "Generation is offline. Start the backend first (Kaggle Run All), then retry.";
  }

  const response = await fetch(`${backend.url}/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    // chat_id lets the backend push the finished PNG directly via sendPhoto.
    body: JSON.stringify({ ...payload, chat_id: chatId }),
    signal: AbortSignal.timeout(10000),
  });

  if (!response.ok) {
    return "The backend rejected the generation request.";
  }
  await response.json();
  return "🖌 Generating your image — it will arrive here shortly.";
}

async function registerBackend(request, env) {
  if (request.headers.get("authorization") !== `Bearer ${env.REGISTRATION_SECRET}`) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }

  const body = await request.json();
  let url;
  try {
    url = new URL(body.url);
  } catch {
    return Response.json({ error: "invalid URL" }, { status: 400 });
  }

  if (url.protocol !== "https:") {
    return Response.json({ error: "HTTPS required" }, { status: 400 });
  }

  const ttl = Math.min(
    Math.max(Number(body.ttl_seconds || DEFAULT_BACKEND_TTL), 60),
    21600,
  );
  const record = {
    url: url.origin,
    device: typeof body.device === "string" ? body.device : "unknown",
    expires_at: Date.now() + ttl * 1000,
  };

  await env.BACKEND_KV.put(BACKEND_KEY, JSON.stringify(record), {
    expirationTtl: ttl,
  });
  return Response.json({ ok: true, ...record });
}

// Handle a tapped glass button.
async function handleCallback(env, ctx, cq) {
  const chatId = cq.message?.chat?.id;
  const data = cq.data || "";

  if (!allowedUser(env, cq.from?.id)) {
    ctx.waitUntil(answerCallback(env, cq.id, "Access denied."));
    return;
  }

  if (data === "gen") {
    // Arm a "waiting for prompt" state; the next plain message becomes the prompt.
    ctx.waitUntil(env.BACKEND_KV.put(`pending:${chatId}`, "1", { expirationTtl: PENDING_TTL }));
    ctx.waitUntil(answerCallback(env, cq.id));
    ctx.waitUntil(reply(env, chatId,
      "✍️ Send me your prompt as a message now (e.g. `1girl, masterpiece, best quality`)."));
    return;
  }

  if (data === "status") {
    const backend = await activeBackend(env);
    const dev = backend?.device && backend.device !== "unknown" ? ` (${backend.device.toUpperCase()})` : "";
    ctx.waitUntil(answerCallback(env, cq.id));
    ctx.waitUntil(reply(env, chatId,
      backend ? `✅ Backend is online${dev}.` : "⚠️ Backend is offline. Start it, then retry.",
      mainMenu(env, backend)));
    return;
  }

  if (data.startsWith("dev:")) {
    const target = data.slice(4);
    const backend = await activeBackend(env);
    if (!backend) {
      ctx.waitUntil(answerCallback(env, cq.id, "Backend is offline."));
      return;
    }
    try {
      const resp = await fetch(`${backend.url}/device`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ device: target }),
        signal: AbortSignal.timeout(15000),
      });
      const body = await resp.json();
      if (!resp.ok) {
        ctx.waitUntil(answerCallback(env, cq.id, body.detail || "Switch failed."));
        return;
      }
      // Refresh the cached device so the menu label flips.
      backend.device = body.device;
      ctx.waitUntil(env.BACKEND_KV.put(BACKEND_KEY, JSON.stringify(backend),
        { expirationTtl: Math.max(60, Math.floor((backend.expires_at - Date.now()) / 1000)) }));
      ctx.waitUntil(answerCallback(env, cq.id, `Now running on ${body.device.toUpperCase()}`));
      ctx.waitUntil(reply(env, chatId,
        body.device === "cpu"
          ? "🐢 Switched to CPU. Free, no GPU quota — but each image takes minutes."
          : "⚡ Switched to GPU. Fast, but uses Kaggle GPU quota.",
        mainMenu(env, backend)));
    } catch {
      ctx.waitUntil(answerCallback(env, cq.id, "Switch timed out."));
    }
    return;
  }

  if (data.startsWith("wake:")) {
    const device = data.slice(5) === "gpu" ? "gpu" : "cpu";
    // If a backend is already live, no need to wake.
    const existing = await activeBackend(env);
    if (existing) {
      ctx.waitUntil(answerCallback(env, cq.id, "Backend already online."));
      ctx.waitUntil(reply(env, chatId, "✅ Backend is already online.", mainMenu(env, existing)));
      return;
    }
    if (!kaggleConfigured(env)) {
      ctx.waitUntil(answerCallback(env, cq.id, "Kaggle API not configured."));
      return;
    }
    ctx.waitUntil(answerCallback(env, cq.id, `Waking on ${device.toUpperCase()}…`));
    ctx.waitUntil((async () => {
      try {
        const { status, body } = await kaggleWake(env, device);
        if (status >= 200 && status < 300) {
          await reply(env, chatId,
            `🚀 Kaggle run started on ${device.toUpperCase()}. It takes ~3-5 min to download the model and register.\n` +
            `Tap 📊 Status in a few minutes; when online you can generate.`);
        } else {
          await reply(env, chatId,
            `Wake failed (${status}): ${body?.message || JSON.stringify(body).slice(0, 200)}`);
        }
      } catch (e) {
        await reply(env, chatId, `Wake error: ${String(e).slice(0, 200)}`);
      }
    })());
    return;
  }

  if (data === "stop") {
    const backend = await activeBackend(env);
    if (!backend) {
      ctx.waitUntil(answerCallback(env, cq.id, "No live backend."));
      ctx.waitUntil(env.BACKEND_KV.delete(BACKEND_KEY));
      return;
    }
    ctx.waitUntil(answerCallback(env, cq.id, "Stopping…"));
    ctx.waitUntil((async () => {
      try {
        await fetch(`${backend.url}/shutdown`, {
          method: "POST",
          signal: AbortSignal.timeout(10000),
        });
      } catch {
        // The process may exit before responding — that's expected.
      }
      // Drop the registration so the menu flips back to Wake buttons.
      await env.BACKEND_KV.delete(BACKEND_KEY);
      await reply(env, chatId, "🛑 Session stopping. The Kaggle run will end shortly, freeing quota.",
        mainMenu(env, null));
    })());
    return;
  }

  ctx.waitUntil(answerCallback(env, cq.id));
}

async function handleMessage(env, ctx, message) {
  const chatId = message.chat.id;
  if (!allowedUser(env, message.from?.id)) {
    ctx.waitUntil(reply(env, chatId, "Access denied."));
    return;
  }

  const text = message.text || "";
  const { command, argument } = parseCommand(text);

  // Not a command? If we're waiting for a prompt (glass Generate was tapped),
  // treat this message as the prompt.
  if (!command.startsWith("/")) {
    const pending = await env.BACKEND_KV.get(`pending:${chatId}`);
    if (pending) {
      ctx.waitUntil(env.BACKEND_KV.delete(`pending:${chatId}`));
      const msg = await submitGeneration(env, chatId, text);
      ctx.waitUntil(reply(env, chatId, msg));
    }
    return;
  }

  if (command === "/start" || command === "/help") {
    const backend = await activeBackend(env);
    ctx.waitUntil(reply(env, chatId,
      "Welcome! Tap 🎨 Generate and then send your prompt, or use /generate <prompt> directly.",
      mainMenu(env, backend)));
    return;
  }

  if (command === "/status") {
    const backend = await activeBackend(env);
    const dev = backend?.device && backend.device !== "unknown" ? ` (${backend.device.toUpperCase()})` : "";
    ctx.waitUntil(reply(env, chatId,
      backend ? `✅ Backend is online${dev}.` : "⚠️ Backend is offline. Start it, then retry.",
      mainMenu(env, backend)));
    return;
  }

  if (command === "/generate") {
    const msg = await submitGeneration(env, chatId, argument);
    ctx.waitUntil(reply(env, chatId, msg));
    return;
  }

  if (command === "/wake") {
    const device = argument.trim().toLowerCase() === "gpu" ? "gpu" : "cpu";
    if (!kaggleConfigured(env)) {
      ctx.waitUntil(reply(env, chatId, "Kaggle API is not configured on this Worker."));
      return;
    }
    const existing = await activeBackend(env);
    if (existing) {
      ctx.waitUntil(reply(env, chatId, "✅ Backend is already online.", mainMenu(env, existing)));
      return;
    }
    ctx.waitUntil((async () => {
      try {
        const { status, body } = await kaggleWake(env, device);
        await reply(env, chatId,
          status >= 200 && status < 300
            ? `🚀 Kaggle run started on ${device.toUpperCase()}. ~3-5 min to be ready; tap 📊 Status.`
            : `Wake failed (${status}): ${body?.message || JSON.stringify(body).slice(0, 200)}`);
      } catch (e) {
        await reply(env, chatId, `Wake error: ${String(e).slice(0, 200)}`);
      }
    })());
    return;
  }

  if (command === "/stop") {
    const backend = await activeBackend(env);
    if (!backend) {
      ctx.waitUntil(env.BACKEND_KV.delete(BACKEND_KEY));
      ctx.waitUntil(reply(env, chatId, "No live backend to stop.", mainMenu(env, null)));
      return;
    }
    ctx.waitUntil((async () => {
      try {
        await fetch(`${backend.url}/shutdown`, { method: "POST", signal: AbortSignal.timeout(10000) });
      } catch {}
      await env.BACKEND_KV.delete(BACKEND_KEY);
      await reply(env, chatId, "🛑 Session stopping — quota freed.", mainMenu(env, null));
    })());
    return;
  }

  ctx.waitUntil(reply(env, chatId, "Unknown command. Use /help.", mainMenu(env, null)));
}

async function webhook(request, env, ctx) {
  if (
    request.headers.get("x-telegram-bot-api-secret-token") !==
    env.TELEGRAM_WEBHOOK_SECRET
  ) {
    return new Response("unauthorized", { status: 401 });
  }

  const update = await request.json();

  if (update.callback_query) {
    await handleCallback(env, ctx, update.callback_query);
    return Response.json({ ok: true });
  }

  const message = update.message;
  if (message?.text) {
    await handleMessage(env, ctx, message);
  }
  return Response.json({ ok: true });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/register-backend") {
      return registerBackend(request, env);
    }
    if (request.method === "POST" && url.pathname === "/telegram/webhook") {
      return webhook(request, env, ctx);
    }
    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ ok: true });
    }
    return new Response("not found", { status: 404 });
  },
};
