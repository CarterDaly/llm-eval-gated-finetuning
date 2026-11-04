"""Tests for LLMService (Ollama/Qwen2.5 integration)."""
import pickle
from unittest.mock import MagicMock, patch

import pytest

from gated_finetune.app import create_app
from gated_finetune.models.prediction import Explanation, Explanations
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.prediction import ModelSQL
from gated_finetune.services.errors import LLMError
from gated_finetune.services.explainer_service import explainer_service
from gated_finetune.services.llm_service import LLMService, llm_service
from gated_finetune.services.mlflow_service import mlflow_service
from gated_finetune.services.umap_service import umap_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyModel:
    feature_names_in_ = ["feature_1", "feature_2"]

    def predict(self, _inputs):
        return ["yes"]


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    yield app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client


def _create_model(app, tmp_path, name="demo", version="1"):
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(DummyModel(), f)

    with app.app_context():
        model_sql = ModelSQL(
            id=f"model-{name}-{version}",
            model_type="generic",
            model_name=name,
            model_version=version,
            artifact_path=str(model_path),
            feature_names=["feature_1", "feature_2"],
            mlflow_run_id="run-1",
        )
        db.session.add(model_sql)
        db.session.commit()

    return model_path


def _fake_shap(_model, _prediction):
    return Explanations(
        explanations=[
            Explanation(name="shap_values", values={"feature_1": 0.1, "feature_2": -0.1})
        ]
    )


def _fake_embedding(_model, _prediction, _config):
    return {"kind": "inputs", "x": 0.5, "y": -0.5}


def _fake_mlflow(**_kwargs):
    return "shap/demo/1/first_inference.json"


# ---------------------------------------------------------------------------
# Unit tests: LLMService.find_nearest_predictions
# ---------------------------------------------------------------------------

class _FakePrediction:
    def __init__(self, id_, label, embeddings):
        self.id = id_
        self.label = label
        self.embeddings = embeddings
        self.features = {"feature_1": 1, "feature_2": 2}


def test_find_nearest_predictions_returns_closest():
    service = LLMService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    close = _FakePrediction("a", "yes", {"x": 0.99, "y": 0.01})
    far = _FakePrediction("b", "no", {"x": -1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [target, close, far], n=1)

    assert len(nearest) == 1
    assert nearest[0].id == "a"


def test_find_nearest_predictions_excludes_self():
    service = LLMService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    other = _FakePrediction("a", "yes", {"x": 1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [target, other], n=5)

    ids = [p.id for p in nearest]
    assert "t" not in ids


def test_find_nearest_predictions_empty_when_no_embeddings():
    service = LLMService()

    target = _FakePrediction("t", "yes", None)
    other = _FakePrediction("a", "yes", {"x": 1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [other], n=5)

    assert nearest == []


def test_find_nearest_predictions_empty_when_target_zero_norm():
    service = LLMService()

    target = _FakePrediction("t", "yes", {"x": 0.0, "y": 0.0})
    other = _FakePrediction("a", "yes", {"x": 1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [other], n=5)

    assert nearest == []


def test_find_nearest_predictions_skips_neighbor_with_no_embeddings():
    service = LLMService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    none_embed = _FakePrediction("a", "yes", None)
    good_neighbor = _FakePrediction("b", "yes", {"x": 0.9, "y": 0.1})

    nearest = service.find_nearest_predictions(target, [none_embed, good_neighbor], n=5)

    ids = [p.id for p in nearest]
    assert "a" not in ids
    assert "b" in ids


def test_find_nearest_predictions_skips_zero_norm_neighbor():
    service = LLMService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    zero_neighbor = _FakePrediction("a", "yes", {"x": 0.0, "y": 0.0})
    good_neighbor = _FakePrediction("b", "yes", {"x": 0.9, "y": 0.1})

    nearest = service.find_nearest_predictions(target, [zero_neighbor, good_neighbor], n=5)

    ids = [p.id for p in nearest]
    assert "a" not in ids
    assert "b" in ids


def test_find_nearest_predictions_respects_n():
    service = LLMService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    others = [_FakePrediction(str(i), "yes", {"x": float(i), "y": 0.0}) for i in range(10)]

    nearest = service.find_nearest_predictions(target, others, n=3)

    assert len(nearest) == 3


# ---------------------------------------------------------------------------
# Unit tests: LLMService.generate_summary
# ---------------------------------------------------------------------------

def test_build_prompt_with_nearest_predictions():
    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})
    near1 = _FakePrediction("a", "no", {"x": 0.9})
    near2 = _FakePrediction("b", "yes", {"x": 0.8})

    prompt = service._build_prompt(target, [near1, near2])

    assert "2 most similar" in prompt
    assert "no" in prompt
    assert "differences" in prompt.lower()


def test_build_prompt_without_nearest_predictions():
    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    prompt = service._build_prompt(target, [])

    assert "No similar" in prompt
    assert "differences" not in prompt.lower()


def _make_mock_response(content="This is a summary."):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": content}}
    return mock_resp


def test_generate_summary_returns_string():
    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with patch("gated_finetune.services.llm_service.requests.post", return_value=_make_mock_response("This is a summary.")):
        result = service.generate_summary(target, [])

    assert result == "This is a summary."


def test_generate_summary_with_nearest_predictions():
    """Covers the if-nearest_predictions branch inside _build_prompt."""
    service = LLMService()
    near = _FakePrediction("n", "no", {"x": 0.9})
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with patch("gated_finetune.services.llm_service.requests.post", return_value=_make_mock_response("Summary with context.")):
        result = service.generate_summary(target, [near])

    assert result == "Summary with context."


def test_generate_summary_wraps_connection_error():
    """A connection failure to Ollama is wrapped in LLMError."""
    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with patch("gated_finetune.services.llm_service.requests.post", side_effect=RuntimeError("connection refused")):
        with pytest.raises(LLMError, match="LLM API call failed"):
            service.generate_summary(target, [])


def test_generate_summary_wraps_exception():
    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with patch("gated_finetune.services.llm_service.requests.post", side_effect=RuntimeError("network error")):
        with pytest.raises(LLMError, match="LLM API call failed"):
            service.generate_summary(target, [])


def test_generate_summary_reraises_llm_error():
    """Covers the 'except LLMError: raise' re-raise branch."""
    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with patch("gated_finetune.services.llm_service.requests.post", side_effect=LLMError("inner LLM error")):
        with pytest.raises(LLMError, match="inner LLM error"):
            service.generate_summary(target, [])


def test_generate_summary_raises_on_http_error():
    """HTTP error status (e.g. 500) is wrapped in LLMError."""
    import requests as req

    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req.HTTPError("500 Server Error")

    with patch("gated_finetune.services.llm_service.requests.post", return_value=mock_resp):
        with pytest.raises(LLMError, match="LLM API call failed"):
            service.generate_summary(target, [])


def test_generate_summary_uses_env_vars(monkeypatch):
    """OLLAMA_BASE_URL and OLLAMA_MODEL env vars are forwarded to the request."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-host:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")

    service = LLMService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with patch("gated_finetune.services.llm_service.requests.post", return_value=_make_mock_response()) as mock_post:
        service.generate_summary(target, [])

    call_args = mock_post.call_args
    assert "http://custom-host:11434/api/chat" in call_args[0][0]
    assert call_args[1]["json"]["model"] == "custom-model"


# ---------------------------------------------------------------------------
# Integration tests: GET /predictions/<id>/explainer
# ---------------------------------------------------------------------------

def test_explainer_endpoint_not_found(client):
    response = client.get("/predictions/nonexistent-id/explainer")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Prediction not found"


def test_explainer_endpoint_returns_summary(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="explainer-demo", version="1")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)

    create_resp = client.post(
        "/predictions/model/explainer-demo/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    assert create_resp.status_code == 201
    prediction_id = create_resp.get_json()["prediction_id"]

    monkeypatch.setattr(
        llm_service,
        "generate_summary",
        lambda prediction, nearest: "The model predicted yes with high confidence.",
    )

    response = client.get(f"/predictions/{prediction_id}/explainer")
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction_id"] == prediction_id
    assert body["label"] == "yes"
    assert "shap_values" in body
    assert "embeddings" in body
    assert body["summary"] == "The model predicted yes with high confidence."


def test_explainer_endpoint_llm_error_returns_500(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="explainer-fail", version="2")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)

    create_resp = client.post(
        "/predictions/model/explainer-fail/version/2",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    assert create_resp.status_code == 201
    prediction_id = create_resp.get_json()["prediction_id"]

    def raise_llm_error(prediction, nearest):
        raise LLMError("LLM API call failed: timeout")

    monkeypatch.setattr(llm_service, "generate_summary", raise_llm_error)

    response = client.get(f"/predictions/{prediction_id}/explainer")
    assert response.status_code == 500
    assert "LLM API call failed" in response.get_json()["error"]


def test_explainer_finds_nearest_from_same_model(app, client, tmp_path, monkeypatch):
    """Nearest prediction search should scope to the same model."""
    _create_model(app, tmp_path, name="explainer-nearest", version="3")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)

    for _ in range(2):
        client.post(
            "/predictions/model/explainer-nearest/version/3",
            json={"features": {"feature_1": 10, "feature_2": 20}},
        )

    captured = {}

    def capture_summary(prediction, nearest):
        captured["nearest_count"] = len(nearest)
        return "Summary."

    monkeypatch.setattr(llm_service, "generate_summary", capture_summary)

    last_resp = client.post(
        "/predictions/model/explainer-nearest/version/3",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    pid = last_resp.get_json()["prediction_id"]

    client.get(f"/predictions/{pid}/explainer")

    assert captured.get("nearest_count", 0) >= 2
