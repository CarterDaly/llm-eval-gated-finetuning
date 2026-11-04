import sys
from unittest.mock import patch

import pytest

from gated_finetune.services.errors import MlflowLogError
from gated_finetune.services.mlflow_service import MlflowService


class _RecordingClient:
    def __init__(self):
        self.logged = []

    def log_dict(self, run_id, payload, artifact_path):
        self.logged.append((run_id, payload, artifact_path))


class _BoomClient:
    def log_dict(self, *a, **kw):
        raise RuntimeError("network error")


def _fake_mlflow(client):
    class _Tracking:
        def MlflowClient(self):
            return client

    class _Mlflow:
        tracking = _Tracking()

    return _Mlflow()


@pytest.fixture
def tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.local")


def _shap_kwargs(**overrides):
    kwargs = dict(
        run_id="run-1",
        model_name="model",
        model_version="1",
        shap_values={"a": 0.1},
        prediction_id="pred-1",
    )
    kwargs.update(overrides)
    return kwargs


def _embedding_kwargs(**overrides):
    kwargs = dict(
        run_id="run-1",
        model_name="m",
        model_version="1",
        embedding={"x": 0.1, "y": 0.2},
        prediction_id="pred-1",
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# log_shap_once
# ---------------------------------------------------------------------------

def test_log_shap_once_skips_when_existing_artifact_path():
    result = MlflowService().log_shap_once(
        **_shap_kwargs(existing_artifact_path="shap/model/1/first_inference.json")
    )
    assert result is None


def test_log_shap_once_requires_run_id(tracking_uri):
    with pytest.raises(MlflowLogError, match="run id"):
        MlflowService().log_shap_once(**_shap_kwargs(run_id=""))


def test_log_shap_once_requires_tracking_uri():
    """Without MLFLOW_TRACKING_URI, MLflow would silently write to ./mlflow.db."""
    with pytest.raises(MlflowLogError, match="MLFLOW_TRACKING_URI"):
        MlflowService().log_shap_once(**_shap_kwargs())


def test_log_shap_once_logs_artifact(monkeypatch, tracking_uri):
    service = MlflowService()
    client = _RecordingClient()
    monkeypatch.setattr(service, "_import_mlflow", lambda: _fake_mlflow(client))

    artifact_path = service.log_shap_once(**_shap_kwargs())

    assert artifact_path == "shap/model/1/first_inference.json"
    run_id, payload, logged_path = client.logged[0]
    assert run_id == "run-1"
    assert logged_path == artifact_path
    assert payload["prediction_id"] == "pred-1"
    assert payload["shap_values"] == {"a": 0.1}


def test_log_shap_once_raises_on_log_dict_exception(monkeypatch, tracking_uri):
    service = MlflowService()
    monkeypatch.setattr(service, "_import_mlflow", lambda: _fake_mlflow(_BoomClient()))

    with pytest.raises(MlflowLogError, match="Failed to log SHAP"):
        service.log_shap_once(**_shap_kwargs())


def test_import_mlflow_not_installed():
    service = MlflowService()
    blocked = {k: None for k in list(sys.modules.keys()) if k == "mlflow" or k.startswith("mlflow.")}
    blocked.setdefault("mlflow", None)
    with patch.dict(sys.modules, blocked):
        with pytest.raises(MlflowLogError, match="not installed"):
            service._import_mlflow()


# ---------------------------------------------------------------------------
# log_embedding_once
# ---------------------------------------------------------------------------

def test_log_embedding_once_skips_when_existing_path():
    result = MlflowService().log_embedding_once(
        **_embedding_kwargs(existing_artifact_path="embeddings/m/1/first_inference.json")
    )
    assert result is None


def test_log_embedding_once_raises_without_run_id(tracking_uri):
    with pytest.raises(MlflowLogError, match="run id is missing"):
        MlflowService().log_embedding_once(**_embedding_kwargs(run_id=""))


def test_log_embedding_once_returns_artifact_path(monkeypatch, tracking_uri):
    service = MlflowService()
    client = _RecordingClient()
    monkeypatch.setattr(service, "_import_mlflow", lambda: _fake_mlflow(client))

    result = service.log_embedding_once(
        **_embedding_kwargs(model_name="my-model", model_version="2")
    )

    assert result == "embeddings/my-model/2/first_inference.json"
    assert client.logged[0][1]["embedding"] == {"x": 0.1, "y": 0.2}


def test_log_embedding_once_wraps_mlflow_exception(monkeypatch, tracking_uri):
    service = MlflowService()
    monkeypatch.setattr(service, "_import_mlflow", lambda: _fake_mlflow(_BoomClient()))

    with pytest.raises(MlflowLogError, match="Failed to log embedding"):
        service.log_embedding_once(**_embedding_kwargs())
