"""
API tests via FastAPI TestClient. Loads the model once per module.
"""
import os
import sys

import pytest


@pytest.fixture(scope="module")
def client(project_root):
    """Spin up the FastAPI app with TestClient. Triggers model load (slow on CPU)."""
    sys.path.insert(0, str(project_root))
    os.environ.setdefault("ENSEMBLE_CONFIG", str(project_root / "models" / "ensemble_final" / "config.json"))
    from fastapi.testclient import TestClient
    from app.api import app, load_model
    load_model()  # explicit, since TestClient may not trigger startup events depending on version
    return TestClient(app)


@pytest.mark.gpu
@pytest.mark.integration
def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_food(client):
    r = client.post("/predict", json={"text": "الاكل بايخ ومالح"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "جودة الطعام"
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_3"]) == 3


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_delivery(client):
    r = client.post("/predict", json={"text": "المندوب تاخر ساعتين"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in {"التوصيل", "وقت الانتظار"}, body  # both reasonable


@pytest.mark.integration
def test_predict_too_short(client):
    r = client.post("/predict", json={"text": "ا"})
    assert r.status_code == 400
