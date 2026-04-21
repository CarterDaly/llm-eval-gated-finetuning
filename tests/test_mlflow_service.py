import sys
from unittest.mock import patch

import pytest

from data5580_hw.services.errors import MlflowLogError
from data5580_hw.services.mlflow_service import MlflowService


def test_log_shap_once_skips_when_existing_artifact_path():
    service = MlflowService()
    result = service.log_shap_once(
        run_id="run-1",
        model_name="model",
        model_version="1",
        shap_values={"a": 0.1},
        prediction_id="pred-1",
        existing_artifact_path="shap/model/1/first_inference.json",
    )
    assert result is None


def test_log_shap_once_requires_run_id():
    service = MlflowService()
    try:
        service.log_shap_once(
            run_id="",
            model_name="model",
            model_version="1",
            shap_values={"a": 0.1},
            prediction_id="pred-1",
        )
    except MlflowLogError as exc:
        assert "run id" in str(exc)
    else:
        raise AssertionError("Expected MlflowLogError for missing run_id.")


def test_log_shap_once_logs_artifact(monkeypatch):
    service = MlflowService()

    class DummyClient:
        def __init__(self):
            self.logged = []

        def log_dict(self, run_id, payload, artifact_path):
            self.logged.append((run_id, payload, artifact_path))

    class DummyTracking:
        def __init__(self, client):
            self._client = client

        def MlflowClient(self):
            return self._client

    class DummyMlflow:
        def __init__(self):
            self.client = DummyClient()
            self.tracking = DummyTracking(self.client)
            self.tracking_uri = None

        def set_tracking_uri(self, uri):
            self.tracking_uri = uri

    dummy = DummyMlflow()
    monkeypatch.setattr(service, "_import_mlflow", lambda: dummy)

    artifact_path = service.log_shap_once(
        run_id="run-1",
        model_name="model",
        model_version="1",
        shap_values={"a": 0.1},
        prediction_id="pred-1",
        tracking_uri="http://mlflow.local",
    )

    assert artifact_path == "shap/model/1/first_inference.json"
    assert dummy.tracking_uri == "http://mlflow.local"
    assert dummy.client.logged
    run_id, payload, logged_path = dummy.client.logged[0]
    assert run_id == "run-1"
    assert logged_path == artifact_path
    assert payload["prediction_id"] == "pred-1"


def test_log_shap_once_raises_on_log_dict_exception(monkeypatch):
    service = MlflowService()

    class _BoomClient:
        def log_dict(self, *a, **kw):
            raise RuntimeError("network error")

    class _BoomTracking:
        def MlflowClient(self):
            return _BoomClient()

    class _BoomMlflow:
        tracking = _BoomTracking()
        def set_tracking_uri(self, uri): pass

    monkeypatch.setattr(service, "_import_mlflow", lambda: _BoomMlflow())

    with pytest.raises(MlflowLogError, match="Failed to log SHAP"):
        service.log_shap_once(
            run_id="run-1",
            model_name="model",
            model_version="1",
            shap_values={"a": 0.1},
            prediction_id="pred-1",
        )


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

def test_log_embedding_once_skips_when_existing_path(monkeypatch):
    """Should return None immediately if an artifact path already exists."""
    service = MlflowService()
    result = service.log_embedding_once(
        run_id="run-1",
        model_name="m",
        model_version="1",
        embedding={"x": 0.1, "y": 0.2},
        prediction_id="pred-1",
        existing_artifact_path="embeddings/m/1/first_inference.json",
    )
    assert result is None


def test_log_embedding_once_raises_without_run_id():
    service = MlflowService()
    with pytest.raises(MlflowLogError, match="run id is missing"):
        service.log_embedding_once(
            run_id="",
            model_name="m",
            model_version="1",
            embedding={"x": 0.1},
            prediction_id="pred-1",
        )


def test_log_embedding_once_returns_artifact_path(monkeypatch):
    service = MlflowService()

    class _FakeClient:
        def log_dict(self, run_id, payload, path):
            pass

    class _FakeTracking:
        def MlflowClient(self):
            return _FakeClient()

    class _FakeMlflow:
        tracking = _FakeTracking()
        def set_tracking_uri(self, uri): pass

    monkeypatch.setattr(service, "_import_mlflow", lambda: _FakeMlflow())

    result = service.log_embedding_once(
        run_id="run-1",
        model_name="my-model",
        model_version="2",
        embedding={"x": 0.5, "y": -0.3},
        prediction_id="pred-abc",
    )
    assert result == "embeddings/my-model/2/first_inference.json"


def test_log_embedding_once_wraps_mlflow_exception(monkeypatch):
    service = MlflowService()

    class _BoomClient:
        def log_dict(self, *a, **kw):
            raise RuntimeError("timeout")

    class _BoomTracking:
        def MlflowClient(self):
            return _BoomClient()

    class _BoomMlflow:
        tracking = _BoomTracking()
        def set_tracking_uri(self, uri): pass

    monkeypatch.setattr(service, "_import_mlflow", lambda: _BoomMlflow())

    with pytest.raises(MlflowLogError, match="Failed to log embedding"):
        service.log_embedding_once(
            run_id="run-1",
            model_name="m",
            model_version="1",
            embedding={"x": 0.1},
            prediction_id="pred-1",
        )
