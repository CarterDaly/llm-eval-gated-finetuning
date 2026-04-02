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
