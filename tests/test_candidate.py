"""Integration tests for the candidate scoring endpoint."""
import pickle

import pytest

from data5580_hw.app import create_app
from data5580_hw.models.prediction import Explanation, Explanations
from data5580_hw.services.connectors.github_connector import GitHubConnectorError, github_connector
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ModelSQL, PredictionSQL
from data5580_hw.services.errors import (
    ArizeLogError,
    EmbeddingError,
    ExplanationError,
    MlflowLogError,
    ModelLoadError,
)
from data5580_hw.services.exploration_service import exploration_service
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.mlflow_service import mlflow_service
from data5580_hw.services.umap_service import umap_service
from data5580_hw.services.arize_service import arize_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "account_age_days", "followers", "public_repos", "stars_received",
    "language_diversity", "recency_days", "avg_repo_stars",
    "forked_ratio", "has_description_ratio", "watchers_received",
]

FAKE_FEATURES = {name: float(i + 1) for i, name in enumerate(FEATURE_NAMES)}


class DummyScoringModel:
    feature_names_in_ = FEATURE_NAMES

    def predict(self, _inputs):
        return [0.75]


@pytest.fixture
def app():
    application = create_app()
    application.config["TESTING"] = True
    yield application


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


def _create_model(app, tmp_path, name="hiring-model", version="1"):
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(DummyScoringModel(), f)

    with app.app_context():
        model_sql = ModelSQL(
            id=f"model-{name}-{version}",
            model_type="generic",
            model_name=name,
            model_version=version,
            artifact_path=str(model_path),
            feature_names=FEATURE_NAMES,
            mlflow_run_id="run-1",
        )
        db.session.add(model_sql)
        db.session.commit()

    return model_path


def _fake_shap(_model, _prediction):
    return Explanations(
        explanations=[Explanation(name="shap_values", values={k: 0.1 for k in FEATURE_NAMES})]
    )


def _fake_embedding(_model, _prediction, _config):
    return {"kind": "inputs", "x": 0.5, "y": -0.5}


def _fake_mlflow(**_kwargs):
    return "shap/hiring/1/path.json"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def test_score_candidate_missing_body_returns_400(client):
    response = client.post("/candidates/torvalds/score", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_score_candidate_missing_required_fields_returns_400(client):
    response = client.post("/candidates/torvalds/score", json={})
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_score_candidate_invalid_umap_returns_400(client):
    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "m", "model_version": "1", "role_context": "r",
              "umap": {"n_components": 1}},
    )
    assert response.status_code == 400
    assert "UMAP" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# GitHub connector errors
# ---------------------------------------------------------------------------

def test_score_candidate_github_error_returns_502(client, monkeypatch):
    monkeypatch.setattr(
        github_connector,
        "extract_features",
        lambda *a: (_ for _ in ()).throw(GitHubConnectorError("rate limit")),
    )
    response = client.post(
        "/candidates/ghost/score",
        json={"model_name": "m", "model_version": "1", "role_context": "Backend engineer"},
    )
    assert response.status_code == 502
    assert "rate limit" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Model not found / load error
# ---------------------------------------------------------------------------

def test_score_candidate_model_not_found_returns_404(client, monkeypatch):
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "ghost-model", "model_version": "99", "role_context": "r"},
    )
    assert response.status_code == 404
    assert response.get_json()["error"] == "Model not found"


def test_score_candidate_model_load_error_returns_500(app, client, monkeypatch):
    with app.app_context():
        model_sql = ModelSQL(
            id="model-broken-cand",
            model_type="generic",
            model_name="broken-cand",
            model_version="1",
            artifact_path="missing.pkl",
            feature_names=FEATURE_NAMES,
            mlflow_run_id=None,
        )
        db.session.add(model_sql)
        db.session.commit()

    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "broken-cand", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Missing feature validation
# ---------------------------------------------------------------------------

def test_score_candidate_missing_features_returns_400(app, client, tmp_path, monkeypatch):
    """If GitHub connector doesn't return all expected features, return 400."""
    _create_model(app, tmp_path, name="feat-check", version="1")

    # Return features missing most of the expected vector
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: {"followers": 10.0})

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "feat-check", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 400
    assert "Missing required features" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Pipeline error cases
# ---------------------------------------------------------------------------

def test_score_candidate_shap_error_returns_400(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="shap-fail-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(
        explainer_service, "create_explanation",
        lambda *a: (_ for _ in ()).throw(ExplanationError("shap failed")),
    )

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "shap-fail-cand", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 400
    assert "shap failed" in response.get_json()["error"]


def test_score_candidate_mlflow_error_returns_500(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="mflow-fail-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(
        mlflow_service, "log_shap_once",
        lambda **kw: (_ for _ in ()).throw(MlflowLogError("mlflow down")),
    )

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "mflow-fail-cand", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 500
    assert "mlflow down" in response.get_json()["error"]


def test_score_candidate_embedding_error_returns_400(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="embed-fail-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(
        umap_service, "create_embedding",
        lambda *a: (_ for _ in ()).throw(EmbeddingError("umap failed")),
    )

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "embed-fail-cand", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 400
    assert "umap failed" in response.get_json()["error"]


def test_score_candidate_db_failure_returns_500(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="db-fail-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)

    original_commit = db.session.commit
    call_count = {"n": 0}

    def flaky_commit():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("DB is down")
        return original_commit()

    monkeypatch.setattr(db.session, "commit", flaky_commit)

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "db-fail-cand", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 500
    assert "Failed to store prediction" in response.get_json()["error"]


def test_score_candidate_arize_error_nonfatal(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="arize-fail-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(
        arize_service, "log_inference",
        lambda **kw: (_ for _ in ()).throw(ArizeLogError("arize down")),
    )

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "arize-fail-cand", "model_version": "1", "role_context": "r"},
    )
    # Non-fatal — should still return 201
    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_score_candidate_success(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="ok-model", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(arize_service, "log_inference", lambda **kw: True)

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "ok-model", "model_version": "1", "role_context": "Senior backend engineer"},
    )

    assert response.status_code == 201
    body = response.get_json()
    assert body["candidate_id"] == "torvalds"
    assert "prediction_id" in body
    assert "fit_score" in body
    assert "shap_values" in body
    assert "embeddings" in body
    assert "features" in body
    assert "exploration_epsilon" in body
    assert 0.0 <= body["exploration_epsilon"] <= 1.0


def test_score_candidate_persists_candidate_id_and_role(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="persist-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(arize_service, "log_inference", lambda **kw: True)

    response = client.post(
        "/candidates/octocat/score",
        json={"model_name": "persist-cand", "model_version": "1",
              "role_context": "ML engineer, Python expertise"},
    )
    assert response.status_code == 201
    prediction_id = response.get_json()["prediction_id"]

    with app.app_context():
        pred = db.session.query(PredictionSQL).filter_by(id=prediction_id).first()
        assert pred is not None
        assert pred.candidate_id == "octocat"
        assert pred.role_context == "ML engineer, Python expertise"


def test_score_candidate_with_umap_config(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="umap-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(arize_service, "log_inference", lambda **kw: True)

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "umap-cand", "model_version": "1", "role_context": "r",
              "umap": {"n_neighbors": 10, "min_dist": 0.2, "n_components": 2}},
    )
    assert response.status_code == 201


def test_score_candidate_mlflow_artifact_path_stored(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="shap-store-cand", version="1")
    monkeypatch.setattr(github_connector, "extract_features", lambda *a: FAKE_FEATURES)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(mlflow_service, "log_shap_once", lambda **kw: "shap/path/stored.json")
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(arize_service, "log_inference", lambda **kw: True)

    response = client.post(
        "/candidates/torvalds/score",
        json={"model_name": "shap-store-cand", "model_version": "1", "role_context": "r"},
    )
    assert response.status_code == 201

    with app.app_context():
        model_sql = db.session.query(ModelSQL).filter_by(model_name="shap-store-cand").first()
        assert model_sql.shap_artifact_path == "shap/path/stored.json"


# ---------------------------------------------------------------------------
# controller helpers
# ---------------------------------------------------------------------------

def test_to_float_score_numeric():
    from data5580_hw.controllers.candidate import _to_float_score
    assert _to_float_score(0.75) == pytest.approx(0.75)
    assert _to_float_score(1.5) == pytest.approx(1.0)  # clamped
    assert _to_float_score(-0.1) == pytest.approx(0.0)  # clamped


def test_to_float_score_non_numeric():
    from data5580_hw.controllers.candidate import _to_float_score
    assert _to_float_score("yes") == pytest.approx(0.5)
    assert _to_float_score(None) == pytest.approx(0.5)


def test_extract_shap_empty():
    from data5580_hw.controllers.candidate import _extract_shap
    assert _extract_shap(None) == {}


def test_extract_shap_wrong_name():
    from data5580_hw.controllers.candidate import _extract_shap
    expl = Explanations(explanations=[Explanation(name="other", values={"f": 0.1})])
    assert _extract_shap(expl) == {}


def test_extract_shap_found():
    from data5580_hw.controllers.candidate import _extract_shap
    expl = Explanations(
        explanations=[Explanation(name="shap_values", values={"account_age_days": 0.3})]
    )
    assert _extract_shap(expl)["account_age_days"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Models: CandidateScoreRequest
# ---------------------------------------------------------------------------

def test_candidate_score_request_valid():
    from data5580_hw.models.candidate import CandidateScoreRequest
    req = CandidateScoreRequest(
        model_name="hiring", model_version="1",
        role_context="Backend engineer with Python experience"
    )
    assert req.umap is None


def test_candidate_score_request_missing_role_context():
    from data5580_hw.models.candidate import CandidateScoreRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CandidateScoreRequest(model_name="m", model_version="1")


# ---------------------------------------------------------------------------
# Prediction schema: candidate fields round-trip
# ---------------------------------------------------------------------------

def test_prediction_candidate_fields():
    from data5580_hw.models.prediction import Prediction
    p = Prediction(
        features={"f": 1},
        candidate_id="octocat",
        role_context="Senior engineer",
    )
    assert p.candidate_id == "octocat"
    assert p.role_context == "Senior engineer"


def test_prediction_candidate_fields_default_none():
    from data5580_hw.models.prediction import Prediction
    p = Prediction(features={"f": 1})
    assert p.candidate_id is None
    assert p.role_context is None
