import pickle

import pytest

from data5580_hw.app import create_app
from data5580_hw.models.prediction import Explanation, Explanations
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.errors import ExplanationError
from data5580_hw.services.explainer_service import explainer_service


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
        )
        db.session.add(model_sql)
        db.session.commit()

    return model_path


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

    response = client.post(
        "/predictions/model/demo/version/1",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 201
    body = response.get_json()
    assert body["label"] == "yes"
    assert body["shap_values"]["feature_1"] == 0.35
    assert body["shap_values"]["feature_2"] == -0.12

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


def test_create_prediction_shap_error_returns_400(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="bad-model", version="2")

    def fake_explainer(_model, _prediction):
        raise ExplanationError("Unsupported model type for SHAP.")

    monkeypatch.setattr(explainer_service, "create_explanation", fake_explainer)

    response = client.post(
        "/predictions/model/bad-model/version/2",
        json={"features": {"feature_1": 10, "feature_2": 20}},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Unsupported model type for SHAP."
