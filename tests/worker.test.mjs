import test from "node:test";
import assert from "node:assert/strict";
import { allowedUser, parseCommand, validateGenerate, miniAppButton } from "../worker/src/index.js";

test("parses commands and bot suffixes", () => {
  assert.deepEqual(parseCommand(" /generate@image_bot a blue fox "), {
    command: "/generate",
    argument: "a blue fox",
  });
});

test("optional Telegram allow-list is enforced", () => {
  assert.equal(allowedUser({}, 100), true);
  assert.equal(
    allowedUser({ ALLOWED_TELEGRAM_USER_IDS: "100, 200" }, 200),
    true,
  );
  assert.equal(
    allowedUser({ ALLOWED_TELEGRAM_USER_IDS: "100, 200" }, 300),
    false,
  );
});

test("generation defaults and bounds", () => {
  const payload = validateGenerate({ prompt: "a blue fox" });
  assert.equal(payload.width, 1024);
  assert.equal(payload.height, 1024);
  assert.equal(payload.steps, 25);
  assert.equal(payload.guidance_scale, 7);
  assert.equal(payload.ip_adapter_scale, 0.6);
  assert.equal(payload.seed, -1);

  assert.throws(
    () => validateGenerate({ prompt: " " }),
    /prompt is required/,
  );
  assert.throws(
    () => validateGenerate({ prompt: "x".repeat(1001) }),
    /prompt is too long/,
  );
  assert.throws(
    () => validateGenerate({ prompt: "fox", steps: 41 }),
    /between 10 and 40/,
  );
  assert.throws(
    () => validateGenerate({ prompt: "fox", width: 2048 }),
    /between 512 and 1024/,
  );
  assert.throws(
    () => validateGenerate({ prompt: "fox", ip_adapter_scale: 1.1 }),
    /between 0 and 1/,
  );
});

test("mini app button builds a web_app pointing at /app", () => {
  const btn = miniAppButton({ url: "https://abc.trycloudflare.com" });
  const kb = btn.reply_markup.inline_keyboard[0][0];
  assert.equal(kb.web_app.url, "https://abc.trycloudflare.com/app");
  assert.match(kb.text, /Web App/);
});

test("mini app button is empty without a live backend", () => {
  assert.deepEqual(miniAppButton(null), {});
  assert.deepEqual(miniAppButton({}), {});
});
