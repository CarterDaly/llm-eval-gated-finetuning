import pickle

import pytest

from data5580_hw.app import create_app
from data5580_hw.models.prediction import Explanation, Explanations, Prediction
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.errors import OpenAIError
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.mlflow_service import mlflow_service
from data5580_hw.services.openai_service import OpenAIService, openai_service
from data5580_hw.services.umap_service import umap_service


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
# Unit tests: OpenAIService.find_nearest_predictions
# ---------------------------------------------------------------------------

class _FakePrediction:
    def __init__(self, id_, label, embeddings):
        self.id = id_
        self.label = label
        self.embeddings = embeddings
        self.features = {"feature_1": 1, "feature_2": 2}


def test_find_nearest_predictions_returns_closest():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    close = _FakePrediction("a", "yes", {"x": 0.99, "y": 0.01})
    far = _FakePrediction("b", "no", {"x": -1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [target, close, far], n=1)

    assert len(nearest) == 1
    assert nearest[0].id == "a"


def test_find_nearest_predictions_excludes_self():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    other = _FakePrediction("a", "yes", {"x": 1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [target, other], n=5)

    ids = [p.id for p in nearest]
    assert "t" not in ids


def test_find_nearest_predictions_empty_when_no_embeddings():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", None)
    other = _FakePrediction("a", "yes", {"x": 1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [other], n=5)

    assert nearest == []


def test_find_nearest_predictions_empty_when_target_zero_norm():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", {"x": 0.0, "y": 0.0})
    other = _FakePrediction("a", "yes", {"x": 1.0, "y": 0.0})

    nearest = service.find_nearest_predictions(target, [other], n=5)

    assert nearest == []


def test_find_nearest_predictions_skips_neighbor_with_no_embeddings():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    none_embed = _FakePrediction("a", "yes", None)
    good_neighbor = _FakePrediction("b", "yes", {"x": 0.9, "y": 0.1})

    nearest = service.find_nearest_predictions(target, [none_embed, good_neighbor], n=5)

    ids = [p.id for p in nearest]
    assert "a" not in ids
    assert "b" in ids


def test_find_nearest_predictions_skips_zero_norm_neighbor():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    zero_neighbor = _FakePrediction("a", "yes", {"x": 0.0, "y": 0.0})
    good_neighbor = _FakePrediction("b", "yes", {"x": 0.9, "y": 0.1})

    nearest = service.find_nearest_predictions(target, [zero_neighbor, good_neighbor], n=5)

    ids = [p.id for p in nearest]
    assert "a" not in ids
    assert "b" in ids


def test_find_nearest_predictions_respects_n():
    service = OpenAIService()

    target = _FakePrediction("t", "yes", {"x": 1.0, "y": 0.0})
    others = [_FakePrediction(str(i), "yes", {"x": float(i), "y": 0.0}) for i in range(10)]

    nearest = service.find_nearest_predictions(target, others, n=3)

    assert len(nearest) == 3


# ---------------------------------------------------------------------------
# Unit tests: OpenAIService.generate_summary
# ---------------------------------------------------------------------------

def test_build_prompt_with_nearest_predictions():
    service = OpenAIService()
    target = _FakePrediction("t", "yes", {"x": 1.0})
    near1 = _FakePrediction("a", "no", {"x": 0.9})
    near2 = _FakePrediction("b", "yes", {"x": 0.8})

    prompt = service._build_prompt(target, [near1, near2])

    assert "2 most similar" in prompt
    assert "no" in prompt
    assert "differences" in prompt.lower()


def test_build_prompt_without_nearest_predictions():
    service = OpenAIService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    prompt = service._build_prompt(target, [])

    assert "No similar" in prompt
    assert "differences" not in prompt.lower()


def test_generate_summary_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = OpenAIService()
    target = _FakePrediction("t", "yes", {"x": 1.0})

    with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
        service.generate_summary(target, [])


def test_generate_summary_returns_string(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "This is a summary."

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = fake_client

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "openai", mock_openai)
        service = OpenAIService()
        target = _FakePrediction("t", "yes", {"x": 1.0})
        result = service.generate_summary(target, [])

    assert result == "This is a summary."


def test_generate_summary_with_nearest_predictions(monkeypatch):
    """Covers the if-nearest_predictions branch inside _build_prompt."""
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.choices[0].message.content = "Summary with context."

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = fake_client

    near = _FakePrediction("n", "no", {"x": 0.9})

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "openai", mock_openai)
        service = OpenAIService()
        target = _FakePrediction("t", "yes", {"x": 1.0})
        result = service.generate_summary(target, [near])

    assert result == "Summary with context."


def test_generate_summary_reraises_openai_error(monkeypatch):
    """Covers the 'except OpenAIError: raise' re-raise branch."""
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_openai = MagicMock()
    mock_openai.OpenAI.side_effect = OpenAIError("inner OpenAI error")

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "openai", mock_openai)
        service = OpenAIService()
        target = _FakePrediction("t", "yes", {"x": 1.0})

        with pytest.raises(OpenAIError, match="inner OpenAI error"):
            service.generate_summary(target, [])


def test_generate_summary_wraps_openai_exception(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("network error")

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = fake_client

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "openai", mock_openai)
        service = OpenAIService()
        target = _FakePrediction("t", "yes", {"x": 1.0})

        with pytest.raises(OpenAIError, match="OpenAI API call failed"):
            service.generate_summary(target, [])


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
        openai_service,
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


def test_explainer_endpoint_openai_error_returns_500(app, client, tmp_path, monkeypatch):
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

    def raise_openai_error(prediction, nearest):
        raise OpenAIError("OpenAI API call failed: timeout")

    monkeypatch.setattr(openai_service, "generate_summary", raise_openai_error)

    response = client.get(f"/predictions/{prediction_id}/explainer")
    assert response.status_code == 500
    assert "OpenAI API call failed" in response.get_json()["error"]


def test_explainer_finds_nearest_from_same_model(app, client, tmp_path, monkeypatch):
    """Nearest prediction search should scope to the same model."""
    _create_model(app, tmp_path, name="explainer-nearest", version="3")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)

    # Create two predictions for the same model
    for _ in range(2):
        client.post(
            "/predictions/model/explainer-nearest/version/3",
            json={"features": {"feature_1": 10, "feature_2": 20}},
        )

    captured = {}

    def capture_summary(prediction, nearest):
        captured["nearest_count"] = len(nearest)
        return "Summary."

    monkeypatch.setattr(openai_service, "generate_summary", capture_summary)

    preds = client.get("/predictions/model/explainer-nearest/version/3") if False else None

    # Get the most recently created prediction id from the create call
    last_resp = client.post(
        "/predictions/model/explainer-nearest/version/3",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    pid = last_resp.get_json()["prediction_id"]

    client.get(f"/predictions/{pid}/explainer")

    # Should have found the 2 earlier predictions as neighbors
    assert captured.get("nearest_count", 0) >= 2
