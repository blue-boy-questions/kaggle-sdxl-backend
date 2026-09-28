import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


APP_PATH = Path(__file__).parents[1] / "backend" / "app.py"
spec = importlib.util.spec_from_file_location("backend_app", APP_PATH)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_health_reports_unloaded_model():
    with TestClient(backend.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "model_loaded": False,
            "queue_size": 0,
        }


def test_generate_reports_backend_unavailable_without_weights():
    with TestClient(backend.app) as client:
        response = client.post("/generate", json={"prompt": "a blue fox"})
        assert response.status_code == 503
        assert response.json()["detail"] == "model pipeline is not loaded"


def test_generation_parameters_are_bounded():
    with TestClient(backend.app) as client:
        response = client.post(
            "/generate",
            json={"prompt": "a blue fox", "steps": 41},
        )
        assert response.status_code == 422

        response = client.post(
            "/generate",
            json={"prompt": "a blue fox", "width": 513},
        )
        assert response.status_code == 422

        response = client.post(
            "/generate",
            json={"prompt": " ", "height": 1024},
        )
        assert response.status_code == 422


def test_missing_job_endpoints_return_404():
    with TestClient(backend.app) as client:
        assert client.get("/status/missing").status_code == 404
        assert client.get("/result/missing").status_code == 404


def test_generate_accepts_optional_chat_id():
    # chat_id is an optional delivery hint; it must validate and default to None.
    assert backend.GenerateRequest(prompt="a fox").chat_id is None
    assert backend.GenerateRequest(prompt="a fox", chat_id=12345).chat_id == 12345


def test_deliver_to_telegram_is_noop_without_token(monkeypatch, tmp_path):
    # With no bot token configured the delivery helper must return silently and
    # never attempt a network call.
    monkeypatch.setattr(backend, "TELEGRAM_BOT_TOKEN", "")
    png = tmp_path / "x.png"
    png.write_bytes(b"fake")
    # Should not raise even though the file/network are not real.
    backend.deliver_to_telegram(123, png, "caption")


def test_deliver_to_telegram_sends_photo_and_schedules_deletion(monkeypatch, tmp_path):
    # With a token set, delivery must: upload the photo, post an English
    # warning, and schedule BOTH the photo and the warning for deletion.
    monkeypatch.setattr(backend, "TELEGRAM_BOT_TOKEN", "TESTTOKEN")

    calls = []
    # Return distinct message_ids per call so we can assert both are scheduled.
    ids = iter([111, 222])

    class FakeResp:
        def __init__(self, mid):
            self._mid = mid

        def json(self):
            return {"result": {"message_id": self._mid}}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResp(next(ids))

    started = {}

    class FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            started["target"] = target
            started["args"] = args

        def start(self):
            started["started"] = True

    import requests as _requests
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setattr(backend.threading, "Thread", FakeThread)

    png = tmp_path / "x.png"
    png.write_bytes(b"fake")
    backend.deliver_to_telegram(123, png, "a fox")

    methods = [url.rsplit("/", 1)[-1] for url, _ in calls]
    assert "sendPhoto" in methods
    assert "sendMessage" in methods
    # Both the photo (111) and the warning (222) are scheduled for deletion.
    assert started.get("started") is True
    assert started["args"] == ([111, 222],)


def test_health_reports_device():
    with TestClient(backend.app) as client:
        body = client.get("/health").json()
        assert "device" in body
        assert body["device"] in ("cpu", "cuda")


def test_device_switch_requires_loaded_model():
    # With no pipeline loaded, /device returns 503.
    with TestClient(backend.app) as client:
        resp = client.post("/device", json={"device": "cpu"})
        assert resp.status_code == 503


def test_device_request_rejects_bad_value():
    with TestClient(backend.app) as client:
        # 'tpu' is not a permitted device -> 422 from the pattern validator.
        assert client.post("/device", json={"device": "tpu"}).status_code == 422
