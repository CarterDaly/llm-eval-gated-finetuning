"""Tests for ArizeService – targets 98%+ coverage of arize_service.py."""
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data5580_hw.services.arize_service import ArizeService
from data5580_hw.services.errors import ArizeLogError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_arize(status_code: int = 200, response_text: str = "OK"):
    """Return (Client_cls, ModelTypes, Environments) mocks."""

    class _Environments:
        PRODUCTION = "production"
        STAGING = "staging"
        TRAINING = "training"

    class _ModelTypes:
        SCORE_CATEGORICAL = "score_categorical"

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = response_text

    mock_client_instance = MagicMock()
    mock_client_instance.log.return_value = mock_response

    mock_client_cls = MagicMock(return_value=mock_client_instance)

    return mock_client_cls, _ModelTypes, _Environments, mock_client_instance


@pytest.fixture
def service():
    return ArizeService()


@pytest.fixture
def arize_env(monkeypatch):
    """Set required Arize env vars."""
    monkeypatch.setenv("ARIZE_API_KEY", "test-api-key")
    monkeypatch.setenv("ARIZE_SPACE_KEY", "test-space-key")
    monkeypatch.setenv("ARIZE_ENABLED", "true")
    monkeypatch.delenv("ARIZE_ENVIRONMENT", raising=False)


# ---------------------------------------------------------------------------
# _is_enabled
# ---------------------------------------------------------------------------

def test_is_enabled_default_true(service, monkeypatch):
    monkeypatch.delenv("ARIZE_ENABLED", raising=False)
    assert service._is_enabled() is True


def test_is_enabled_explicit_true(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "true")
    assert service._is_enabled() is True


def test_is_enabled_false_string(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "false")
    assert service._is_enabled() is False


def test_is_enabled_zero_string(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "0")
    assert service._is_enabled() is False


def test_is_enabled_no_string(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "no")
    assert service._is_enabled() is False


# ---------------------------------------------------------------------------
# _get_credentials
# ---------------------------------------------------------------------------

def test_get_credentials_success(service, monkeypatch):
    monkeypatch.setenv("ARIZE_API_KEY", "my-api-key")
    monkeypatch.setenv("ARIZE_SPACE_KEY", "my-space-key")
    api_key, space_key = service._get_credentials()
    assert api_key == "my-api-key"
    assert space_key == "my-space-key"


def test_get_credentials_missing_api_key(service, monkeypatch):
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    monkeypatch.setenv("ARIZE_SPACE_KEY", "my-space-key")
    with pytest.raises(ArizeLogError, match="ARIZE_API_KEY"):
        service._get_credentials()


def test_get_credentials_missing_space_key(service, monkeypatch):
    monkeypatch.setenv("ARIZE_API_KEY", "my-api-key")
    monkeypatch.delenv("ARIZE_SPACE_KEY", raising=False)
    with pytest.raises(ArizeLogError, match="ARIZE_SPACE_KEY"):
        service._get_credentials()


# ---------------------------------------------------------------------------
# _resolve_environment
# ---------------------------------------------------------------------------

class _Envs:
    PRODUCTION = "production"
    STAGING = "staging"
    TRAINING = "training"


def test_resolve_environment_production():
    result = ArizeService._resolve_environment(_Envs, "production")
    assert result == _Envs.PRODUCTION


def test_resolve_environment_staging():
    result = ArizeService._resolve_environment(_Envs, "staging")
    assert result == _Envs.STAGING


def test_resolve_environment_development():
    result = ArizeService._resolve_environment(_Envs, "development")
    assert result == _Envs.TRAINING


def test_resolve_environment_unknown_defaults_to_production():
    result = ArizeService._resolve_environment(_Envs, "unknown-env")
    assert result == _Envs.PRODUCTION


def test_resolve_environment_case_insensitive():
    result = ArizeService._resolve_environment(_Envs, "STAGING")
    assert result == _Envs.STAGING


# ---------------------------------------------------------------------------
# _import_arize
# ---------------------------------------------------------------------------

def test_import_arize_sdk_not_installed(service):
    """_import_arize raises ArizeLogError when the Arize SDK cannot be imported."""
    from unittest.mock import patch

    blocked = {k: None for k in list(sys.modules.keys()) if "arize" in k}
    blocked.setdefault("arize", None)
    blocked.setdefault("arize.api", None)
    blocked.setdefault("arize.utils", None)
    blocked.setdefault("arize.utils.types", None)

    with patch.dict(sys.modules, blocked):
        with pytest.raises(ArizeLogError, match="not installed"):
            service._import_arize()


def test_import_arize_success(service, monkeypatch):
    """_import_arize should return three objects when SDK is present."""
    mock_client = MagicMock()
    mock_model_types = MagicMock()
    mock_environments = MagicMock()

    mock_arize_api = MagicMock()
    mock_arize_api.Client = mock_client
    mock_arize_types = MagicMock()
    mock_arize_types.ModelTypes = mock_model_types
    mock_arize_types.Environments = mock_environments

    monkeypatch.setitem(sys.modules, "arize", MagicMock())
    monkeypatch.setitem(sys.modules, "arize.api", mock_arize_api)
    monkeypatch.setitem(sys.modules, "arize.utils", MagicMock())
    monkeypatch.setitem(sys.modules, "arize.utils.types", mock_arize_types)

    Client, ModelTypes, Environments = service._import_arize()
    assert Client is mock_client
    assert ModelTypes is mock_model_types
    assert Environments is mock_environments


# ---------------------------------------------------------------------------
# log_inference
# ---------------------------------------------------------------------------

def test_log_inference_disabled_returns_false(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "false")
    result = service.log_inference(
        prediction_id="p1",
        model_name="m",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
    )
    assert result is False


def test_log_inference_success(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    result = service.log_inference(
        prediction_id="p1",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"feature_1": 10, "feature_2": 20},
    )

    assert result is True
    mock_client_instance.log.assert_called_once()
    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["prediction_id"] == "p1"
    assert call_kwargs["model_id"] == "demo"
    assert call_kwargs["model_version"] == "1"
    assert call_kwargs["prediction_label"] == "yes"
    assert call_kwargs["features"] == {"feature_1": 10, "feature_2": 20}


def test_log_inference_with_actual_label(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    service.log_inference(
        prediction_id="p2",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
        actual_label="no",
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["actual_label"] == "no"


def test_log_inference_with_shap_values(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    service.log_inference(
        prediction_id="p3",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
        shap_values={"f": 0.5},
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["shap_values"] == {"f": 0.5}


def test_log_inference_with_timestamp(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    service.log_inference(
        prediction_id="p4",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
        timestamp=ts,
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["prediction_timestamp"] == int(ts.timestamp())


def test_log_inference_without_optional_fields(service, monkeypatch, arize_env):
    """No optional kwargs should be present when not supplied."""
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    service.log_inference(
        prediction_id="p5",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert "actual_label" not in call_kwargs
    assert "shap_values" not in call_kwargs
    assert "prediction_timestamp" not in call_kwargs


def test_log_inference_default_environment_is_development(service, monkeypatch, arize_env):
    """When ARIZE_ENVIRONMENT is unset, the default should be development (TRAINING)."""
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))
    # arize_env fixture already deletes ARIZE_ENVIRONMENT

    service.log_inference(
        prediction_id="default-env",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["environment"] == Environments.TRAINING


def test_log_inference_uses_environment_env_var(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))
    monkeypatch.setenv("ARIZE_ENVIRONMENT", "staging")

    service.log_inference(
        prediction_id="p6",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["environment"] == Environments.STAGING


def test_log_inference_environment_parameter_overrides_env_var(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))
    monkeypatch.setenv("ARIZE_ENVIRONMENT", "production")

    service.log_inference(
        prediction_id="p7",
        model_name="demo",
        model_version="1",
        prediction_label="yes",
        features={"f": 1},
        environment="staging",
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["environment"] == Environments.STAGING


def test_log_inference_api_non_200_raises(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize(
        status_code=500, response_text="Internal Server Error"
    )
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    with pytest.raises(ArizeLogError, match="500"):
        service.log_inference(
            prediction_id="p8",
            model_name="demo",
            model_version="1",
            prediction_label="yes",
            features={"f": 1},
        )


def test_log_inference_client_exception_raises(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    mock_client_instance.log.side_effect = RuntimeError("network error")
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    with pytest.raises(ArizeLogError, match="Failed to log inference"):
        service.log_inference(
            prediction_id="p9",
            model_name="demo",
            model_version="1",
            prediction_label="yes",
            features={"f": 1},
        )


def test_log_inference_missing_api_key_raises(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "true")
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    monkeypatch.setenv("ARIZE_SPACE_KEY", "sk")

    with pytest.raises(ArizeLogError, match="ARIZE_API_KEY"):
        service.log_inference(
            prediction_id="p10",
            model_name="demo",
            model_version="1",
            prediction_label="yes",
            features={"f": 1},
        )


def test_log_inference_missing_space_key_raises(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "true")
    monkeypatch.setenv("ARIZE_API_KEY", "ak")
    monkeypatch.delenv("ARIZE_SPACE_KEY", raising=False)

    with pytest.raises(ArizeLogError, match="ARIZE_SPACE_KEY"):
        service.log_inference(
            prediction_id="p11",
            model_name="demo",
            model_version="1",
            prediction_label="yes",
            features={"f": 1},
        )


# ---------------------------------------------------------------------------
# log_actual
# ---------------------------------------------------------------------------

def test_log_actual_disabled_returns_false(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "false")
    result = service.log_actual(
        prediction_id="p1",
        model_name="m",
        model_version="1",
        actual_label="yes",
    )
    assert result is False


def test_log_actual_success(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    result = service.log_actual(
        prediction_id="p1",
        model_name="demo",
        model_version="1",
        actual_label="no",
    )

    assert result is True
    mock_client_instance.log.assert_called_once()
    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["prediction_id"] == "p1"
    assert call_kwargs["model_id"] == "demo"
    assert call_kwargs["actual_label"] == "no"


def test_log_actual_with_environment_parameter(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    service.log_actual(
        prediction_id="p2",
        model_name="demo",
        model_version="1",
        actual_label="yes",
        environment="staging",
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["environment"] == Environments.STAGING


def test_log_actual_api_non_200_raises(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize(
        status_code=400, response_text="Bad Request"
    )
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    with pytest.raises(ArizeLogError, match="400"):
        service.log_actual(
            prediction_id="p3",
            model_name="demo",
            model_version="1",
            actual_label="yes",
        )


def test_log_actual_client_exception_raises(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    mock_client_instance.log.side_effect = ConnectionError("timeout")
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))

    with pytest.raises(ArizeLogError, match="Failed to log actual"):
        service.log_actual(
            prediction_id="p4",
            model_name="demo",
            model_version="1",
            actual_label="yes",
        )


def test_log_actual_missing_credentials_raises(service, monkeypatch):
    monkeypatch.setenv("ARIZE_ENABLED", "true")
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    monkeypatch.delenv("ARIZE_SPACE_KEY", raising=False)

    with pytest.raises(ArizeLogError):
        service.log_actual(
            prediction_id="p5",
            model_name="demo",
            model_version="1",
            actual_label="yes",
        )


def test_log_actual_uses_environment_env_var(service, monkeypatch, arize_env):
    Client, ModelTypes, Environments, mock_client_instance = _make_mock_arize()
    monkeypatch.setattr(service, "_import_arize", lambda: (Client, ModelTypes, Environments))
    monkeypatch.setenv("ARIZE_ENVIRONMENT", "development")

    service.log_actual(
        prediction_id="p6",
        model_name="demo",
        model_version="1",
        actual_label="yes",
    )

    call_kwargs = mock_client_instance.log.call_args.kwargs
    assert call_kwargs["environment"] == Environments.TRAINING
