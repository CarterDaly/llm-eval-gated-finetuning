import pickle

import pytest

from data5580_hw.app import create_app
from data5580_hw.models.prediction import Explanation, Explanations
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.errors import EmbeddingError, ExplanationError, MlflowLogError
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
