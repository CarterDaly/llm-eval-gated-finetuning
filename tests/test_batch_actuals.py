"""Tests for PATCH /predictions/actuals (batch label import)."""
import pickle

import pytest

from gated_finetune.app import create_app
from gated_finetune.models.prediction import Explanation, Explanations
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.prediction import ModelSQL
from gated_finetune.services.arize_service import arize_service
from gated_finetune.services.explainer_service import explainer_service
from gated_finetune.services.mlflow_service import mlflow_service
from gated_finetune.services.umap_service import umap_service


class _DummyModel:
    feature_names_in_ = ["f1", "f2"]

    def predict(self, _x):
        return ["yes"]


def _fake_shap(_m, _p):
    return Explanations(
        explanations=[Explanation(name="shap_values", values={"f1": 0.1, "f2": -0.1})]
    )


def _fake_embed(_m, _p, _c):
    return {"kind": "inputs", "x": 0.1, "y": 0.2}


def _fake_mlflow(**_kw):
    return "shap/path.json"


@pytest.fixture
def app():
    a = create_app()
    a.config["TESTING"] = True
    yield a


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


def _register_model(app, tmp_path, name="batch-model", version="1"):
    path = tmp_path / "m.pkl"
    with open(path, "wb") as f:
        pickle.dump(_DummyModel(), f)
    with app.app_context():
        db.session.add(ModelSQL(
            id=f"model-{name}-{version}",
            model_type="generic",
            model_name=name,
            model_version=version,
            artifact_path=str(path),
            feature_names=["f1", "f2"],
            mlflow_run_id="r1",
        ))
        db.session.commit()


def _create_prediction(client, name="batch-model", version="1"):
    resp = client.post(
        f"/predictions/model/{name}/version/{version}",
        json={"features": {"f1": 1, "f2": 2}},
    )
    assert resp.status_code == 201
    return resp.get_json()["prediction_id"]


@pytest.fixture
def setup(app, client, tmp_path, monkeypatch):
    _register_model(app, tmp_path)
    monkeypatch.setattr(explainer_service, "create_explanation", _fake_shap)
    monkeypatch.setattr(umap_service, "create_embedding", _fake_embed)
    monkeypatch.setattr(mlflow_service, "log_shap_once", _fake_mlflow)
    monkeypatch.setattr(arize_service, "log_actual", lambda **_kw: None)
    yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_batch_update_all_found(setup):
    p1 = _create_prediction(setup)
    p2 = _create_prediction(setup)

    resp = setup.patch(
        "/predictions/actuals",
        json={"updates": [
            {"prediction_id": p1, "actual": "yes"},
            {"prediction_id": p2, "actual": "no"},
        ]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body["updated"]) == {p1, p2}
    assert body["not_found"] == []


def test_batch_update_partial_not_found(setup):
    p1 = _create_prediction(setup)

    resp = setup.patch(
        "/predictions/actuals",
        json={"updates": [
            {"prediction_id": p1, "actual": "yes"},
            {"prediction_id": "nonexistent-id", "actual": "no"},
        ]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert p1 in body["updated"]
    assert "nonexistent-id" in body["not_found"]


def test_batch_update_all_not_found(setup):
    resp = setup.patch(
        "/predictions/actuals",
        json={"updates": [
            {"prediction_id": "ghost-1", "actual": "yes"},
            {"prediction_id": "ghost-2", "actual": "no"},
        ]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["updated"] == []
    assert set(body["not_found"]) == {"ghost-1", "ghost-2"}


def test_batch_update_missing_updates_field(setup):
    resp = setup.patch("/predictions/actuals", json={})
    assert resp.status_code == 400


def test_batch_update_empty_list(setup):
    resp = setup.patch("/predictions/actuals", json={"updates": []})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["updated"] == []
    assert body["not_found"] == []


def test_batch_update_invalid_body(setup):
    resp = setup.patch("/predictions/actuals", json="not a dict")
    assert resp.status_code == 400


def test_batch_update_persists_actual_value(app, setup):
    from gated_finetune.services.database.prediction import PredictionSQL

    p1 = _create_prediction(setup)

    setup.patch(
        "/predictions/actuals",
        json={"updates": [{"prediction_id": p1, "actual": "confirmed"}]},
    )

    with app.app_context():
        row = db.session.query(PredictionSQL).filter(PredictionSQL.id == p1).first()
        assert row.actual == "confirmed"
