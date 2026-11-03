"""Tests for Celery task routes, controller, and task functions."""
import pickle

import pytest

from gated_finetune.app import create_app
from gated_finetune.celery_app import celery, init_celery
from gated_finetune.models.prediction import Explanation, Explanations
from gated_finetune.models.task import TaskStatus
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.prediction import ModelSQL, PredictionSQL
from gated_finetune.services.errors import (
    ArizeLogError,
    EmbeddingError,
    ExplanationError,
    InferenceError,
    MlflowLogError,
    ModelLoadError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyModel:
    feature_names_in_ = ["feature_1", "feature_2"]

    def __init__(self, output="yes"):
        self.output = output

    def predict(self, _inputs):
        return [self.output]


@pytest.fixture
def app():
    application = create_app()
    application.config["TESTING"] = True
    # Configure Celery to run tasks synchronously in tests (no broker needed)
    celery.conf.update(
        task_always_eager=True,
        task_eager_propagates=False,
        result_backend="cache+memory://",
    )
    yield application


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


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


# ---------------------------------------------------------------------------
# Model: TaskStatus
# ---------------------------------------------------------------------------


def test_task_status_model_defaults():
    ts = TaskStatus(task_id="abc", status="PENDING")
    assert ts.task_id == "abc"
    assert ts.status == "PENDING"
    assert ts.result is None
    assert ts.error is None


def test_task_status_model_with_result():
    ts = TaskStatus(task_id="xyz", status="SUCCESS", result={"label": "yes"})
    assert ts.result == {"label": "yes"}
    assert ts.error is None


def test_task_status_model_with_error():
    ts = TaskStatus(task_id="xyz", status="FAILURE", error="something went wrong")
    assert ts.error == "something went wrong"
    assert ts.result is None


# ---------------------------------------------------------------------------
# celery_app: init_celery
# ---------------------------------------------------------------------------


def test_init_celery_sets_flask_app(app):
    assert hasattr(celery, "flask_app")
    assert celery.flask_app is app


def test_init_celery_configures_serializer(app):
    assert celery.conf.task_serializer == "json"
    assert celery.conf.result_serializer == "json"


def test_init_celery_idempotent(app):
    """Calling init_celery again with the same app should not raise."""
    init_celery(app)
    assert celery.flask_app is app


# ---------------------------------------------------------------------------
# Routes: GET /tasks/<task_id>
# ---------------------------------------------------------------------------


def test_get_task_status_unknown_task(client):
    """Unknown task IDs should return PENDING status (Celery default)."""
    response = client.get("/tasks/nonexistent-task-id")
    assert response.status_code == 200
    body = response.get_json()
    assert body["task_id"] == "nonexistent-task-id"
    assert body["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Routes: POST /tasks/predictions (short task)
# ---------------------------------------------------------------------------


def test_submit_short_task_missing_features_returns_400(client):
    response = client.post(
        "/tasks/predictions/model/demo/version/1",
        json={},
        query_string={"task_type": "short"},
    )
    assert response.status_code == 400
    assert "features" in response.get_json()["error"]


def test_submit_short_task_non_json_returns_400(client):
    response = client.post(
        "/tasks/predictions/model/demo/version/1",
        data="not json",
        content_type="text/plain",
        query_string={"task_type": "short"},
    )
    assert response.status_code == 400


def test_submit_short_task_invalid_umap_returns_400(client):
    response = client.post(
        "/tasks/predictions/model/demo/version/1",
        json={"features": {"f": 1}, "umap": {"n_components": 1}},
    )
    assert response.status_code == 400
    assert "UMAP" in response.get_json()["error"]


def test_submit_short_task_invalid_body_returns_400(client):
    response = client.post(
        "/tasks/predictions/model/demo/version/1",
        json={"features": "not-a-dict"},
        query_string={"task_type": "short"},
    )
    assert response.status_code == 400


def test_submit_short_task_success(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="short-ok", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.services import mlflow_service as mf_mod

    monkeypatch.setattr(es_mod.explainer_service, "create_explanation", lambda *a: Explanations())
    monkeypatch.setattr(us_mod.umap_service, "create_embedding", lambda *a: {"kind": "inputs", "x": 0.1, "y": 0.2})
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: None)

    response = client.post(
        "/tasks/predictions/model/short-ok/version/1",
        json={"features": {"feature_1": 1, "feature_2": 2}},
        query_string={"task_type": "short"},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert "task_id" in body
    assert body["status"] in {"SUCCESS", "PENDING", "STARTED"}


def test_submit_short_task_model_not_found(client):
    response = client.post(
        "/tasks/predictions/model/missing/version/99",
        json={"features": {"feature_1": 1}},
        query_string={"task_type": "short"},
    )
    # With eager mode the task runs immediately; model not found → error result
    body = response.get_json()
    # status 201 because task was submitted; result carries the error
    assert body["task_id"] is not None


# ---------------------------------------------------------------------------
# Routes: POST /tasks/predictions (long task, default)
# ---------------------------------------------------------------------------


def test_submit_long_task_success(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-ok", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import arize_service as ar_mod

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.1, "feature_2": -0.1})]
        ),
    )
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: {"kind": "inputs", "x": 0.5, "y": -0.5},
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: "shap/path")
    monkeypatch.setattr(ar_mod.arize_service, "log_inference", lambda **kw: True)

    response = client.post(
        "/tasks/predictions/model/long-ok/version/1",
        json={"features": {"feature_1": 1, "feature_2": 2}},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert "task_id" in body
    assert body["status"] in {"SUCCESS", "PENDING", "STARTED"}


def test_submit_long_task_with_wait(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-wait", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import arize_service as ar_mod

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.2, "feature_2": -0.2})]
        ),
    )
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: {"kind": "inputs", "x": 1.0, "y": -1.0},
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: None)
    monkeypatch.setattr(ar_mod.arize_service, "log_inference", lambda **kw: True)

    response = client.post(
        "/tasks/predictions/model/long-wait/version/1",
        json={"features": {"feature_1": 1, "feature_2": 2}},
        query_string={"wait": "true"},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert "task_id" in body


def test_submit_short_task_with_wait(app, client, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="short-wait", version="1")

    response = client.post(
        "/tasks/predictions/model/short-wait/version/1",
        json={"features": {"feature_1": 1, "feature_2": 2}},
        query_string={"task_type": "short", "wait": "true"},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert "task_id" in body


# ---------------------------------------------------------------------------
# Task function: run_short_prediction (unit tests)
# ---------------------------------------------------------------------------


def test_run_short_prediction_model_not_found(app):
    from gated_finetune.tasks.prediction_tasks import run_short_prediction

    with app.app_context():
        result = run_short_prediction(
            "no-such-model", "1", {"feature_1": 1, "feature_2": 2}
        )
    assert result["error"] == "Model not found"
    assert result["status_code"] == 404


def test_run_short_prediction_model_load_error(app, tmp_path):
    with app.app_context():
        model_sql = ModelSQL(
            id="model-broken-short",
            model_type="generic",
            model_name="broken-short",
            model_version="1",
            artifact_path="nonexistent.pkl",
            feature_names=["feature_1", "feature_2"],
            mlflow_run_id="run-broken",
        )
        db.session.add(model_sql)
        db.session.commit()

    from gated_finetune.tasks.prediction_tasks import run_short_prediction

    with app.app_context():
        result = run_short_prediction("broken-short", "1", {"feature_1": 1, "feature_2": 2})
    assert "status_code" in result
    assert result["status_code"] == 500


def test_run_short_prediction_inference_error(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="short-infer-fail", version="1")

    from gated_finetune.services import model_service as ms_mod
    from gated_finetune.tasks.prediction_tasks import run_short_prediction

    monkeypatch.setattr(ms_mod.model_service, "create_inference", lambda *a: (_ for _ in ()).throw(InferenceError("boom")))

    with app.app_context():
        result = run_short_prediction("short-infer-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "boom"
    assert result["status_code"] == 500


def test_run_short_prediction_db_commit_failure(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="short-db-fail", version="1")

    from gated_finetune.tasks.prediction_tasks import run_short_prediction

    original_commit = db.session.commit
    call_count = {"n": 0}

    def flaky_commit():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("DB down")
        return original_commit()

    monkeypatch.setattr(db.session, "commit", flaky_commit)

    with app.app_context():
        result = run_short_prediction("short-db-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "Failed to store prediction"
    assert result["status_code"] == 500


def test_run_short_prediction_success(app, tmp_path):
    _create_model(app, tmp_path, name="short-success", version="1")

    from gated_finetune.tasks.prediction_tasks import run_short_prediction

    with app.app_context():
        result = run_short_prediction("short-success", "1", {"feature_1": 1, "feature_2": 2})
    assert "prediction_id" in result
    assert result["label"] == "yes"


def test_run_short_prediction_persists_to_db(app, tmp_path):
    _create_model(app, tmp_path, name="short-persist", version="1")

    from gated_finetune.tasks.prediction_tasks import run_short_prediction

    with app.app_context():
        result = run_short_prediction("short-persist", "1", {"feature_1": 1, "feature_2": 2})
        prediction_id = result["prediction_id"]
        pred = db.session.query(PredictionSQL).filter_by(id=prediction_id).first()
        assert pred is not None
        assert pred.label == "yes"


# ---------------------------------------------------------------------------
# Task function: run_long_prediction (unit tests)
# ---------------------------------------------------------------------------


def test_run_long_prediction_model_not_found(app):
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    with app.app_context():
        result = run_long_prediction("ghost-model", "1", {"feature_1": 1})
    assert result["error"] == "Model not found"


def test_run_long_prediction_model_load_error(app):
    with app.app_context():
        model_sql = ModelSQL(
            id="model-long-broken",
            model_type="generic",
            model_name="long-broken",
            model_version="1",
            artifact_path="missing.pkl",
            feature_names=["feature_1"],
            mlflow_run_id=None,
        )
        db.session.add(model_sql)
        db.session.commit()

    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    with app.app_context():
        result = run_long_prediction("long-broken", "1", {"feature_1": 1})
    assert result["status_code"] == 500


def test_run_long_prediction_inference_error(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-infer-fail", version="1")

    from gated_finetune.services import model_service as ms_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        ms_mod.model_service,
        "create_inference",
        lambda *a: (_ for _ in ()).throw(InferenceError("infer boom")),
    )

    with app.app_context():
        result = run_long_prediction("long-infer-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "infer boom"


def test_run_long_prediction_explanation_error(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-shap-fail", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: (_ for _ in ()).throw(ExplanationError("shap boom")),
    )

    with app.app_context():
        result = run_long_prediction("long-shap-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "shap boom"
    assert result["status_code"] == 400


def test_run_long_prediction_mlflow_error(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-mflow-fail", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.0, "feature_2": 0.0})]
        ),
    )
    monkeypatch.setattr(
        mf_mod.mlflow_service,
        "log_shap_once",
        lambda **kw: (_ for _ in ()).throw(MlflowLogError("mlflow boom")),
    )

    with app.app_context():
        result = run_long_prediction("long-mflow-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "mlflow boom"
    assert result["status_code"] == 500


def test_run_long_prediction_embedding_error(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-umap-fail", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.0, "feature_2": 0.0})]
        ),
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: None)
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: (_ for _ in ()).throw(EmbeddingError("umap boom")),
    )

    with app.app_context():
        result = run_long_prediction("long-umap-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "umap boom"
    assert result["status_code"] == 400


def test_run_long_prediction_db_commit_failure(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-db-fail", version="1")

    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.0, "feature_2": 0.0})]
        ),
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: None)
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: {"kind": "inputs", "x": 0.0, "y": 0.0},
    )

    original_commit = db.session.commit
    call_count = {"n": 0}

    def flaky_commit():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("DB down")
        return original_commit()

    monkeypatch.setattr(db.session, "commit", flaky_commit)

    with app.app_context():
        result = run_long_prediction("long-db-fail", "1", {"feature_1": 1, "feature_2": 2})
    assert result["error"] == "Failed to store prediction"


def test_run_long_prediction_arize_error_non_fatal(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-arize-fail", version="1")

    from gated_finetune.services import arize_service as ar_mod
    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.1, "feature_2": -0.1})]
        ),
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: None)
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: {"kind": "inputs", "x": 0.5, "y": -0.5},
    )
    monkeypatch.setattr(
        ar_mod.arize_service,
        "log_inference",
        lambda **kw: (_ for _ in ()).throw(ArizeLogError("arize boom")),
    )

    with app.app_context():
        result = run_long_prediction("long-arize-fail", "1", {"feature_1": 1, "feature_2": 2})
    # Arize failure is non-fatal — prediction should still succeed
    assert "prediction_id" in result
    assert result["label"] == "yes"


def test_run_long_prediction_success_full(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-full", version="1")

    from gated_finetune.services import arize_service as ar_mod
    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.3, "feature_2": -0.3})]
        ),
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: "shap/long/1/path.json")
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: {"kind": "inputs", "x": 0.7, "y": -0.7},
    )
    monkeypatch.setattr(ar_mod.arize_service, "log_inference", lambda **kw: True)

    with app.app_context():
        result = run_long_prediction(
            "long-full", "1", {"feature_1": 1, "feature_2": 2},
            tags={"source": "test"}, actual="no", threshold=0.5,
        )

    assert "prediction_id" in result
    assert result["label"] == "yes"
    assert result["shap_values"]["feature_1"] == pytest.approx(0.3)
    assert result["embeddings"]["x"] == pytest.approx(0.7)


def test_run_long_prediction_with_umap_config(app, tmp_path, monkeypatch):
    _create_model(app, tmp_path, name="long-umap-cfg", version="1")

    from gated_finetune.services import arize_service as ar_mod
    from gated_finetune.services import explainer_service as es_mod
    from gated_finetune.services import mlflow_service as mf_mod
    from gated_finetune.services import umap_service as us_mod
    from gated_finetune.tasks.prediction_tasks import run_long_prediction

    monkeypatch.setattr(
        es_mod.explainer_service,
        "create_explanation",
        lambda *a: Explanations(
            explanations=[Explanation(name="shap_values", values={"feature_1": 0.0, "feature_2": 0.0})]
        ),
    )
    monkeypatch.setattr(mf_mod.mlflow_service, "log_shap_once", lambda **kw: None)
    monkeypatch.setattr(
        us_mod.umap_service,
        "create_embedding",
        lambda *a: {"kind": "inputs", "x": 0.1, "y": 0.2},
    )
    monkeypatch.setattr(ar_mod.arize_service, "log_inference", lambda **kw: True)

    with app.app_context():
        result = run_long_prediction(
            "long-umap-cfg", "1", {"feature_1": 1, "feature_2": 2},
            umap_config={"n_neighbors": 10, "min_dist": 0.2, "n_components": 2},
        )
    assert "prediction_id" in result


# ---------------------------------------------------------------------------
# Task helpers: _extract_shap and _json_safe
# ---------------------------------------------------------------------------


def test_extract_shap_no_explanations():
    from gated_finetune.tasks.prediction_tasks import _extract_shap

    assert _extract_shap(None) == {}


def test_extract_shap_name_not_found():
    from gated_finetune.tasks.prediction_tasks import _extract_shap

    expl = Explanations(explanations=[Explanation(name="other", values={"f": 0.1})])
    assert _extract_shap(expl) == {}


def test_extract_shap_found():
    from gated_finetune.tasks.prediction_tasks import _extract_shap

    expl = Explanations(
        explanations=[Explanation(name="shap_values", values={"f": 0.5})]
    )
    assert _extract_shap(expl) == {"f": 0.5}


def test_json_safe_numpy():
    import numpy as np

    from gated_finetune.tasks.prediction_tasks import _json_safe

    val = np.float32(2.5)
    result = _json_safe(val)
    assert result == pytest.approx(2.5)
    assert not hasattr(result, "item")


def test_json_safe_plain():
    from gated_finetune.tasks.prediction_tasks import _json_safe

    assert _json_safe("hello") == "hello"
    assert _json_safe(42) == 42


# ---------------------------------------------------------------------------
# Controller: _result_response with error in result dict
# ---------------------------------------------------------------------------


def test_result_response_success_with_error_dict():
    """A SUCCESS task that returns {error: ...} should be reported as FAILURE."""
    from unittest.mock import MagicMock

    from gated_finetune.controllers.task import _result_response

    mock_result = MagicMock()
    mock_result.status = "SUCCESS"
    mock_result.result = {"error": "something failed", "status_code": 404}
    mock_result.id = "test-id"

    ts = _result_response(mock_result)
    assert ts.status == "FAILURE"
    assert ts.error == "something failed"


def test_result_response_failure():
    from unittest.mock import MagicMock

    from gated_finetune.controllers.task import _result_response

    mock_result = MagicMock()
    mock_result.status = "FAILURE"
    mock_result.result = Exception("task exploded")
    mock_result.id = "test-id-2"

    ts = _result_response(mock_result)
    assert ts.status == "FAILURE"
    assert "task exploded" in ts.error


def test_result_response_pending():
    from unittest.mock import MagicMock

    from gated_finetune.controllers.task import _result_response

    mock_result = MagicMock()
    mock_result.status = "PENDING"
    mock_result.result = None
    mock_result.id = "test-id-3"

    ts = _result_response(mock_result)
    assert ts.status == "PENDING"
    assert ts.result is None
    assert ts.error is None


# ---------------------------------------------------------------------------
# Controller: submit_prediction – wait=true with exception
# ---------------------------------------------------------------------------


def test_submit_wait_true_get_exception(app, client, tmp_path, monkeypatch):
    """If task.get() raises, submit returns 500."""
    _create_model(app, tmp_path, name="wait-exc", version="1")

    from unittest.mock import MagicMock, patch

    mock_result = MagicMock()
    mock_result.get.side_effect = Exception("broker unreachable")
    mock_result.id = "fake-task-id"

    with patch("gated_finetune.controllers.task.run_long_prediction") as mock_task:
        mock_task.delay.return_value = mock_result
        response = client.post(
            "/tasks/predictions/model/wait-exc/version/1",
            json={"features": {"feature_1": 1, "feature_2": 2}},
            query_string={"wait": "true"},
        )

    assert response.status_code == 500
    assert "broker unreachable" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# celery_app signals (coverage)
# ---------------------------------------------------------------------------


def test_worker_signals_do_not_raise():
    """Ensure signal handlers can be called without raising."""
    from gated_finetune.celery_app import on_task_failure, on_task_retry, on_worker_ready, on_worker_shutdown
    from unittest.mock import MagicMock

    on_worker_ready()
    on_worker_shutdown()
    on_task_failure(task_id="t1", exception=ValueError("oops"))
    req = MagicMock()
    req.id = "t2"
    on_task_retry(request=req, reason=RuntimeError("retry"))
