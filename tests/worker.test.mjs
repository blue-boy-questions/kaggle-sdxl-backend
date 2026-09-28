import test from "node:test";
import assert from "node:assert/strict";
import { allowedUser, parseCommand, validateGenerate, mainMenu } from "../worker/src/index.js";

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

test("main menu shows glass Generate + Status buttons", () => {
  const menu = mainMenu({}, null);
  const rows = menu.reply_markup.inline_keyboard;
  assert.equal(rows[0][0].callback_data, "gen");
  assert.match(rows[0][0].text, /Generate/);
  assert.equal(rows[1][0].callback_data, "status");
});

test("main menu offers Web App + a device toggle when backend is live", () => {
  const menu = mainMenu({}, { url: "https://abc.trycloudflare.com", device: "cuda" });
  const flat = menu.reply_markup.inline_keyboard.flat();
  // Web App button points at /app
  const webapp = flat.find((b) => b.web_app);
  assert.equal(webapp.web_app.url, "https://abc.trycloudflare.com/app");
  // On GPU, the toggle offers switching to CPU
  const toggle = flat.find((b) => b.callback_data && b.callback_data.startsWith("dev:"));
  assert.equal(toggle.callback_data, "dev:cpu");
  assert.match(toggle.text, /CPU/);
});

test("device toggle flips to GPU when on CPU", () => {
  const menu = mainMenu({}, { url: "https://x.trycloudflare.com", device: "cpu" });
  const toggle = menu.reply_markup.inline_keyboard.flat()
    .find((b) => b.callback_data && b.callback_data.startsWith("dev:"));
  assert.equal(toggle.callback_data, "dev:cuda");
  assert.match(toggle.text, /GPU/);
});

test("main menu has no web_app / device toggle without a backend", () => {
  const flat = mainMenu({}, null).reply_markup.inline_keyboard.flat();
  assert.equal(flat.find((b) => b.web_app), undefined);
  assert.equal(flat.find((b) => b.callback_data && b.callback_data.startsWith("dev:")), undefined);
});

test("live backend menu includes a Stop button", () => {
  const flat = mainMenu({}, { url: "https://x.trycloudflare.com", device: "cuda" })
    .reply_markup.inline_keyboard.flat();
  const stop = flat.find((b) => b.callback_data === "stop");
  assert.ok(stop, "expected a Stop session button");
  assert.match(stop.text, /Stop/);
});

test("offline menu shows Wake buttons when Kaggle is configured", () => {
  const env = { KAGGLE_USERNAME: "u", KAGGLE_KEY: "k", KAGGLE_KERNEL: "u/n" };
  const flat = mainMenu(env, null).reply_markup.inline_keyboard.flat();
  const wakeCpu = flat.find((b) => b.callback_data === "wake:cpu");
  const wakeGpu = flat.find((b) => b.callback_data === "wake:gpu");
  assert.ok(wakeCpu && wakeGpu, "expected CPU and GPU wake buttons");
  assert.match(wakeCpu.text, /CPU/);
  assert.match(wakeGpu.text, /GPU/);
});

test("offline menu without Kaggle config shows no wake buttons", () => {
  const flat = mainMenu({}, null).reply_markup.inline_keyboard.flat();
  assert.equal(flat.find((b) => b.callback_data && b.callback_data.startsWith("wake:")), undefined);
});
