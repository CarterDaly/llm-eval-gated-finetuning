import pickle

import pytest

from data5580_hw.app import create_app
from data5580_hw.models.prediction import Explanation, Explanations
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.arize_service import arize_service
from data5580_hw.services.errors import ArizeLogError, EmbeddingError, ExplanationError, InferenceError, MlflowLogError
from data5580_hw.services.model_service import model_service
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.mlflow_service import mlflow_service
from data5580_hw.services.umap_service import umap_service


class DummyModel:
    feature_names_in_ = ["feature_1", "feature_2"]

    def __init__(self, output="yes"):
        self.output = output

    def predict(self, _inputs):
        return [self.output]


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
    with open(model_path, "wb") as handle:
        pickle.dump(DummyModel(), handle)

    with app.app_context():
        model_sql = ModelSQL(
            id="model-1",
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


def _create_broken_model(app, name="broken", version="1"):
    with app.app_context():
        model_sql = ModelSQL(
            id="model-broken",
            model_type="generic",
            model_name=name,
            model_version=version,
            artifact_path="missing.pkl",
            feature_names=["feature_1", "feature_2"],
            mlflow_run_id="run-broken",
        )
        db.session.add(model_sql)
        db.session.commit()


def test_create_prediction_returns_shap_values_and_persists(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path)

    def fake_explainer(_model, _prediction):
        return Explanations(
            explanations=[
                Explanation(
                    name="shap_values",
                    values={"feature_1": 0.35, "feature_2": -0.12},
                )
            ]
        )

    monkeypatch.setattr(explainer_service, "create_explanation", fake_explainer)
    monkeypatch.setattr(
        umap_service,
        "create_embedding",
        lambda _model, _prediction, _config: {"kind": "inputs", "x": 0.24, "y": -0.38},
    )
    monkeypatch.setattr(
        mlflow_service,
        "log_shap_once",
        lambda **_kwargs: "shap/demo/1/first_inference.json",
    )

    response = client.post(
        "/predictions/model/demo/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 201
    body = response.get_json()
    prediction_id = body["prediction_id"]
    assert body["label"] == "yes"
    assert body["shap_values"]["feature_1"] == 0.35
    assert body["shap_values"]["feature_2"] == -0.12
    assert body["embeddings"]["x"] == 0.24
    assert body["embeddings"]["y"] == -0.38

    with app.app_context():
        prediction = db.session.query(PredictionSQL).first()
        assert prediction is not None
        explanation = (
            db.session.query(ExplanationSQL)
            .filter(ExplanationSQL.prediction_id == prediction.id)
            .first()
        )
        assert explanation is not None
        assert explanation.values["feature_1"] == 0.35
        assert prediction.embeddings["x"] == 0.24
        assert prediction.model.id == "model-1"
        model_sql = db.session.query(ModelSQL).first()
        assert model_sql.shap_artifact_path == "shap/demo/1/first_inference.json"

    response = client.get(f"/predictions/{prediction_id}")
    assert response.status_code == 200
    body = response.get_json()
    assert body["embeddings"]["x"] == 0.24


def test_create_prediction_missing_features_returns_400(client):
    response = client.post("/predictions/model/demo/version/1", json={})
    assert response.status_code == 400
    assert "features" in response.get_json()["error"]


def test_create_prediction_model_not_found(client):
    response = client.post(
        "/predictions/model/missing/version/1",
        json={"features": {"feature_1": 1}},
    )
    assert response.status_code == 404
    assert response.get_json()["error"] == "Model not found"


def test_create_prediction_missing_required_feature_returns_400(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="needs-feature", version="1")
    monkeypatch.setattr(
        mlflow_service,
        "log_shap_once",
        lambda **_kwargs: "shap/needs-feature/1/first_inference.json",
    )

    response = client.post(
        "/predictions/model/needs-feature/version/1",
        json={"features": {"feature_1": 10}},
    )

    assert response.status_code == 400
    assert "feature_2" in response.get_json()["error"]


def test_create_prediction_invalid_umap_config_returns_400(client):
    response = client.post(
        "/predictions/model/missing/version/1",
        json={
            "features": {"feature_1": 1},
            "umap": {"n_components": 1},
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid UMAP configuration"


def test_create_prediction_shap_error_returns_400(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="bad-model", version="2")

    def fake_explainer(_model, _prediction):
        raise ExplanationError("Unsupported model type for SHAP.")

    monkeypatch.setattr(explainer_service, "create_explanation", fake_explainer)
    monkeypatch.setattr(
        umap_service,
        "create_embedding",
        lambda _model, _prediction, _config: {"kind": "inputs", "x": 0.0, "y": 0.0},
    )
    monkeypatch.setattr(
        mlflow_service,
        "log_shap_once",
        lambda **_kwargs: "shap/bad-model/2/first_inference.json",
    )

    response = client.post(
        "/predictions/model/bad-model/version/2",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Unsupported model type for SHAP."


def test_create_prediction_model_load_error_returns_500(app, client, monkeypatch):
    _create_broken_model(app, name="broken-model", version="1")

    response = client.post(
        "/predictions/model/broken-model/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 500
    assert "Model artifact not found" in response.get_json()["error"]


def test_create_prediction_umap_error_returns_400(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="umap-fail", version="3")

    def fake_explainer(_model, _prediction):
        return Explanations(
            explanations=[
                Explanation(
                    name="shap_values",
                    values={"feature_1": 0.0, "feature_2": 0.0},
                )
            ]
        )

    def fake_embedding(_model, _prediction, _config):
        raise EmbeddingError("Failed to compute UMAP embedding.")

    monkeypatch.setattr(explainer_service, "create_explanation", fake_explainer)
    monkeypatch.setattr(umap_service, "create_embedding", fake_embedding)
    monkeypatch.setattr(
        mlflow_service,
        "log_shap_once",
        lambda **_kwargs: "shap/umap-fail/3/first_inference.json",
    )

    response = client.post(
        "/predictions/model/umap-fail/version/3",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Failed to compute UMAP embedding."


def test_create_prediction_mlflow_error_returns_500(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="mlflow-fail", version="4")

    def fake_explainer(_model, _prediction):
        return Explanations(
            explanations=[
                Explanation(
                    name="shap_values",
                    values={"feature_1": 0.0, "feature_2": 0.0},
                )
            ]
        )

    def fake_embedding(_model, _prediction, _config):
        return {"kind": "inputs", "x": 0.0, "y": 0.0}

    def fake_mlflow(**_kwargs):
        raise MlflowLogError("Failed to log SHAP values to MLflow.")

    monkeypatch.setattr(explainer_service, "create_explanation", fake_explainer)
    monkeypatch.setattr(umap_service, "create_embedding", fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", fake_mlflow)

    response = client.post(
        "/predictions/model/mlflow-fail/version/4",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to log SHAP values to MLflow."


# ---------------------------------------------------------------------------
# Arize integration – controller-level tests
# ---------------------------------------------------------------------------

def _fake_explainer_shap(model, prediction):
    return Explanations(
        explanations=[
            Explanation(name="shap_values", values={"feature_1": 0.1, "feature_2": -0.1})
        ]
    )


def _fake_embedding(model, prediction, config):
    return {"kind": "inputs", "x": 0.5, "y": -0.5}


def _fake_mlflow(**kwargs):
    return "shap/demo/1/first_inference.json"


def test_create_prediction_calls_arize_log_inference(app, client, tmp_path, monkeypatch):
    """Arize log_inference should be invoked after a successful prediction."""
    _create_model(app, tmp_path, name="arize-demo", version="5")

    calls = []

    def fake_log_inference(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", fake_log_inference)

    response = client.post(
        "/predictions/model/arize-demo/version/5",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 201
    assert len(calls) == 1
    assert calls[0]["model_name"] == "arize-demo"
    assert calls[0]["model_version"] == "5"
    assert calls[0]["prediction_label"] == "yes"
    assert calls[0]["features"] == {"feature_1": 10, "feature_2": 20}


def test_create_prediction_arize_error_is_non_fatal(app, client, tmp_path, monkeypatch):
    """An ArizeLogError must not cause the prediction endpoint to fail."""
    _create_model(app, tmp_path, name="arize-fail", version="6")

    def fake_log_inference(**kwargs):
        raise ArizeLogError("Arize is down.")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", fake_log_inference)

    response = client.post(
        "/predictions/model/arize-fail/version/6",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 201
    assert "prediction_id" in response.get_json()


def test_update_actual_calls_arize_log_actual(app, client, tmp_path, monkeypatch):
    """After updating the actual, Arize log_actual should be called."""
    _create_model(app, tmp_path, name="arize-actual", version="7")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)

    actual_calls = []

    def fake_log_inference(**kwargs):
        return True

    def fake_log_actual(**kwargs):
        actual_calls.append(kwargs)
        return True

    monkeypatch.setattr(arize_service, "log_inference", fake_log_inference)
    monkeypatch.setattr(arize_service, "log_actual", fake_log_actual)

    create_resp = client.post(
        "/predictions/model/arize-actual/version/7",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    assert create_resp.status_code == 201
    prediction_id = create_resp.get_json()["prediction_id"]

    patch_resp = client.patch(
        f"/predictions/{prediction_id}/actual",
        json={"actual": "no"},
    )

    assert patch_resp.status_code == 200
    assert len(actual_calls) == 1
    assert actual_calls[0]["actual_label"] == "no"
    assert actual_calls[0]["model_name"] == "arize-actual"


def test_update_actual_arize_error_is_non_fatal(app, client, tmp_path, monkeypatch):
    """An ArizeLogError from log_actual must not cause the update endpoint to fail."""
    _create_model(app, tmp_path, name="arize-actual-fail", version="8")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", lambda **_: True)

    def fake_log_actual(**kwargs):
        raise ArizeLogError("Arize is unreachable.")

    monkeypatch.setattr(arize_service, "log_actual", fake_log_actual)

    create_resp = client.post(
        "/predictions/model/arize-actual-fail/version/8",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    assert create_resp.status_code == 201
    prediction_id = create_resp.get_json()["prediction_id"]

    patch_resp = client.patch(
        f"/predictions/{prediction_id}/actual",
        json={"actual": "yes"},
    )

    assert patch_resp.status_code == 200
    assert patch_resp.get_json()["message"] == "Prediction updated"


# ---------------------------------------------------------------------------
# Additional prediction controller coverage
# ---------------------------------------------------------------------------

def test_create_prediction_non_json_body_returns_400(client):
    """Non-JSON body triggers the _get_prediction_request non-dict path."""
    response = client.post(
        "/predictions/model/demo/version/1",
        data="not json",
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert "features" in response.get_json()["error"]


def test_create_prediction_invalid_input_generic_returns_400(client):
    """Validation error that is not missing and not umap hits generic branch."""
    # features with an invalid type causes pydantic to fail with a type error,
    # not a "missing" error, and without "umap" in loc → generic branch.
    response = client.post(
        "/predictions/model/demo/version/1",
        json={"features": "not-a-dict"},
    )
    assert response.status_code == 400


def test_create_prediction_legacy_endpoint(app, client, tmp_path, monkeypatch):
    """The /model/<name>/version/<ver> legacy route should work identically."""
    _create_model(app, tmp_path, name="legacy", version="1")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", lambda **_: True)

    response = client.post(
        "/model/legacy/version/1",
        json={"features": {"feature_1": 1, "feature_2": 2}},
    )
    assert response.status_code == 201


def test_get_prediction_not_found_returns_404(client):
    response = client.get("/predictions/nonexistent-id")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Prediction not found"


def test_update_actual_missing_body_returns_400(client):
    response = client.patch("/predictions/any-id/actual", json={"wrong_key": "x"})
    assert response.status_code == 400
    assert "actual" in response.get_json()["error"]


def test_update_actual_not_found_returns_404(client):
    response = client.patch("/predictions/nonexistent/actual", json={"actual": "yes"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Prediction not found"


def test_extract_shap_values_returns_empty_when_no_explanations():
    from data5580_hw.controllers.prediction import _extract_shap_values
    assert _extract_shap_values(None) == {}


def test_extract_shap_values_returns_empty_when_name_not_found():
    from data5580_hw.controllers.prediction import _extract_shap_values
    explanations = Explanations(
        explanations=[Explanation(name="other", values={"f": 0.1})]
    )
    assert _extract_shap_values(explanations) == {}


def test_extract_embeddings_returns_empty_when_none():
    from data5580_hw.controllers.prediction import _extract_embeddings
    from data5580_hw.models.prediction import Prediction
    p = Prediction(features={"f": 1})
    assert _extract_embeddings(p) == {}


def test_json_safe_calls_item_on_numpy_scalar():
    from data5580_hw.controllers.prediction import _json_safe
    import numpy as np
    val = np.float32(1.5)
    result = _json_safe(val)
    assert result == pytest.approx(1.5)
    assert not hasattr(result, "item")


# ---------------------------------------------------------------------------
# Additional controller branch coverage
# ---------------------------------------------------------------------------

class _MinimalModel:
    """A model with no feature_names_in_ so load_model leaves feature_names as None."""
    def predict(self, _inputs):
        return ["yes"]


def _create_model_no_features(app, tmp_path, name="no-feat", version="1"):
    """Create a model record with no feature_names to test the no-feature-names branch."""
    model_path = tmp_path / "model_nf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(_MinimalModel(), f)

    with app.app_context():
        model_sql = ModelSQL(
            id=f"model-nf-{name}-{version}",
            model_type="generic",
            model_name=name,
            model_version=version,
            artifact_path=str(model_path),
            feature_names=None,
            mlflow_run_id="run-nf",
        )
        db.session.add(model_sql)
        db.session.commit()


def test_create_prediction_no_feature_names_skips_validation(app, client, tmp_path, monkeypatch):
    """When a model has no feature_names, _validate_feature_presence returns None (line 110)."""
    _create_model_no_features(app, tmp_path, name="nf-model", version="1")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", lambda **_: True)

    response = client.post(
        "/predictions/model/nf-model/version/1",
        json={"features": {"anything": 99}},
    )

    assert response.status_code == 201


def test_create_prediction_inference_error_returns_500(app, client, tmp_path, monkeypatch):
    """InferenceError from model_service.create_inference should return 500 (lines 120-122, 245)."""
    _create_model(app, tmp_path, name="infer-fail", version="1")

    def fake_inference(model, prediction):
        from data5580_hw.services.errors import InferenceError
        raise InferenceError("Inference exploded.")

    monkeypatch.setattr(model_service, "create_inference", fake_inference)

    response = client.post(
        "/predictions/model/infer-fail/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 500
    assert "Inference exploded." in response.get_json()["error"]


def test_create_prediction_db_commit_failure_returns_500(app, client, tmp_path, monkeypatch):
    """DB commit failure in _persist_prediction should return 500 (lines 207-210, 263)."""
    _create_model(app, tmp_path, name="db-fail", version="1")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", lambda **_: True)

    original_commit = db.session.commit

    call_count = {"n": 0}

    def flaky_commit():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("DB is down")
        return original_commit()

    monkeypatch.setattr(db.session, "commit", flaky_commit)

    response = client.post(
        "/predictions/model/db-fail/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 500
    assert "Failed to store prediction" in response.get_json()["error"]


def test_update_actual_db_commit_failure_returns_500(app, client, tmp_path, monkeypatch):
    """DB commit failure in update_actual should return 500 (lines 362-364)."""
    _create_model(app, tmp_path, name="upd-db-fail", version="1")

    monkeypatch.setattr(explainer_service, "create_explanation", _fake_explainer_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embedding)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_inference", lambda **_: True)

    create_resp = client.post(
        "/predictions/model/upd-db-fail/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )
    assert create_resp.status_code == 201
    prediction_id = create_resp.get_json()["prediction_id"]

    def broken_commit():
        raise Exception("DB is down during update")

    monkeypatch.setattr(db.session, "commit", broken_commit)

    patch_resp = client.patch(
        f"/predictions/{prediction_id}/actual",
        json={"actual": "no"},
    )

    assert patch_resp.status_code == 500
    assert "Failed to update prediction" in patch_resp.get_json()["error"]
