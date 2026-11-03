"""Tests for ExplainerService – targets 95%+ coverage of explainer_service.py."""
import numpy as np
import pytest

from gated_finetune.models.prediction import Explanation, Explanations, Model, Prediction
from gated_finetune.services.errors import ExplanationError
from gated_finetune.services.explainer_service import ExplainerService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loaded_model(name="demo", version="1", feature_names=None):
    """Return a Model with a fake _model already attached."""

    class _FakeSklearn:
        feature_names_in_ = feature_names or ["f1", "f2"]

        def predict(self, X):
            return ["yes"]

    m = Model(name=name, version=version, feature_names=feature_names or ["f1", "f2"])
    m._model = _FakeSklearn()
    return m


def _prediction(features=None):
    return Prediction(features=features or {"f1": 1.0, "f2": 2.0})


def _fake_shap_result(values):
    """Wrap raw array in a shap-like object with .values attribute."""
    class _ShapResult:
        pass
    r = _ShapResult()
    r.values = values
    return r


@pytest.fixture
def service():
    return ExplainerService()


# ---------------------------------------------------------------------------
# create_explanation – model not loaded
# ---------------------------------------------------------------------------

def test_create_explanation_model_not_loaded(service):
    model = Model(name="x", version="1")
    with pytest.raises(ExplanationError, match="not loaded"):
        service.create_explanation(model, _prediction())


# ---------------------------------------------------------------------------
# create_explanation – uses cached explainer on model
# ---------------------------------------------------------------------------

def test_create_explanation_uses_model_cached_explainer(service, monkeypatch):
    """If model._explainer is already set, the service should reuse it."""
    calls = []

    class _CachedExplainer:
        def __call__(self, frame):
            calls.append(True)
            return _fake_shap_result(np.array([[0.1, 0.2]]))

    model = _loaded_model()
    model._explainer = _CachedExplainer()

    result = service.create_explanation(model, _prediction())
    assert len(calls) == 1
    assert isinstance(result, Explanations)


# ---------------------------------------------------------------------------
# create_explanation – shap call raises
# ---------------------------------------------------------------------------

def test_create_explanation_shap_raises(service, monkeypatch):
    class _BoomExplainer:
        def __call__(self, frame):
            raise RuntimeError("shap failed")

    model = _loaded_model()
    model._explainer = _BoomExplainer()

    with pytest.raises(ExplanationError, match="Failed to compute SHAP"):
        service.create_explanation(model, _prediction())


# ---------------------------------------------------------------------------
# _get_or_create_explainer – cache hit
# ---------------------------------------------------------------------------

def test_get_or_create_explainer_cache_hit(service, monkeypatch):
    class _DummyExplainer:
        def __call__(self, frame):
            return _fake_shap_result(np.array([[0.5, -0.5]]))

    model = _loaded_model(name="cached", version="1")
    service._explainer_cache[("cached", "1")] = _DummyExplainer()

    result = service.create_explanation(model, _prediction())
    assert result.explanations[0].name == "shap_values"


# ---------------------------------------------------------------------------
# _get_or_create_explainer – import fails
# ---------------------------------------------------------------------------

def test_get_or_create_explainer_import_fails(service, monkeypatch):
    def _boom():
        raise ExplanationError("SHAP is not installed.")

    monkeypatch.setattr(service, "_import_shap", _boom)
    model = _loaded_model(name="noshap", version="1")

    with pytest.raises(ExplanationError, match="not installed"):
        service.create_explanation(model, _prediction())


# ---------------------------------------------------------------------------
# _get_or_create_explainer – build explainer raises
# ---------------------------------------------------------------------------

def test_get_or_create_explainer_build_raises(service, monkeypatch):
    import shap as real_shap

    def _boom_build(shap_mod, model, frame):
        raise ValueError("unsupported")

    monkeypatch.setattr(service, "_import_shap", lambda: real_shap)
    monkeypatch.setattr(ExplainerService, "_build_explainer", staticmethod(_boom_build))

    model = _loaded_model(name="badbuild", version="1")
    with pytest.raises(ExplanationError, match="Unsupported model type"):
        service.create_explanation(model, _prediction())


# ---------------------------------------------------------------------------
# _import_shap
# ---------------------------------------------------------------------------

def test_import_shap_success(service):
    result = service._import_shap()
    import shap
    assert result is shap


def test_import_shap_not_installed(service, monkeypatch):
    import sys
    from unittest.mock import patch

    blocked = {k: None for k in list(sys.modules.keys()) if k == "shap" or k.startswith("shap.")}
    blocked.setdefault("shap", None)

    with patch.dict(sys.modules, blocked):
        with pytest.raises(ExplanationError, match="not installed"):
            service._import_shap()


# ---------------------------------------------------------------------------
# _build_explainer – model type routing
# ---------------------------------------------------------------------------

def _mock_shap():
    """Return a fake shap module that records which explainer was used."""
    class _FakeExplainer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, frame):
            return _fake_shap_result(np.array([[0.0, 0.0]]))

    class _FakeShap:
        TreeExplainer = _FakeExplainer
        LinearExplainer = _FakeExplainer
        DeepExplainer = _FakeExplainer
        KernelExplainer = _FakeExplainer
        Explainer = _FakeExplainer

    return _FakeShap()


def test_build_explainer_tree_type(service):
    fake_shap = _mock_shap()

    class _TreeModel:
        tree_ = object()

    m = Model(name="t", version="1", type="tree", feature_names=["f1", "f2"])
    m._model = _TreeModel()
    explainer = service._build_explainer(fake_shap, m, None)
    assert explainer is not None


def test_build_explainer_linear_type(service):
    fake_shap = _mock_shap()
    import pandas as pd

    frame = pd.DataFrame([{"f1": 1, "f2": 2}])
    m = Model(name="l", version="1", type="linear", feature_names=["f1", "f2"])
    m._model = object()
    explainer = service._build_explainer(fake_shap, m, frame)
    assert explainer is not None


def test_build_explainer_deep_type(service):
    fake_shap = _mock_shap()
    import pandas as pd

    frame = pd.DataFrame([{"f1": 1, "f2": 2}])
    m = Model(name="d", version="1", type="deep", feature_names=["f1", "f2"])
    m._model = object()
    explainer = service._build_explainer(fake_shap, m, frame)
    assert explainer is not None


def test_build_explainer_kernel_type(service):
    fake_shap = _mock_shap()
    import pandas as pd

    frame = pd.DataFrame([{"f1": 1, "f2": 2}])

    class _M:
        def predict(self, X):
            return [0]

    m = Model(name="k", version="1", type="kernel", feature_names=["f1", "f2"])
    m._model = _M()
    explainer = service._build_explainer(fake_shap, m, frame)
    assert explainer is not None


def test_build_explainer_generic_type(service):
    fake_shap = _mock_shap()
    import pandas as pd

    frame = pd.DataFrame([{"f1": 1, "f2": 2}])
    m = Model(name="g", version="1", type="generic", feature_names=["f1", "f2"])
    m._model = object()
    explainer = service._build_explainer(fake_shap, m, frame)
    assert explainer is not None


# ---------------------------------------------------------------------------
# _to_shap_mapping
# ---------------------------------------------------------------------------

def test_to_shap_mapping_1d_array(service):
    result = service._to_shap_mapping(np.array([0.1, -0.2]), ["f1", "f2"])
    assert result == {"f1": pytest.approx(0.1), "f2": pytest.approx(-0.2)}


def test_to_shap_mapping_2d_array(service):
    arr = np.array([[0.3, 0.4]])
    result = service._to_shap_mapping(arr, ["f1", "f2"])
    assert result["f1"] == pytest.approx(0.3)


def test_to_shap_mapping_3d_array(service):
    arr = np.array([[[0.5, 0.6]]])
    result = service._to_shap_mapping(arr, ["f1", "f2"])
    assert result["f1"] == pytest.approx(0.5)


def test_to_shap_mapping_shap_result_object(service):
    result = service._to_shap_mapping(
        _fake_shap_result(np.array([0.7, 0.8])), ["f1", "f2"]
    )
    assert result["f2"] == pytest.approx(0.8)


def test_to_shap_mapping_list_input(service):
    result = service._to_shap_mapping([np.array([0.1, 0.2])], ["f1", "f2"])
    assert result["f1"] == pytest.approx(0.1)


def test_to_shap_mapping_unsupported_shape_raises(service):
    arr = np.zeros((2, 2, 2, 2))
    with pytest.raises(ExplanationError, match="Unsupported SHAP output shape"):
        service._to_shap_mapping(arr, ["f1", "f2"])


def test_to_shap_mapping_wrong_length_raises(service):
    with pytest.raises(ExplanationError, match="does not match feature count"):
        service._to_shap_mapping(np.array([0.1, 0.2, 0.3]), ["f1", "f2"])


def test_to_shap_mapping_numpy_not_installed_raises(service, monkeypatch):
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"numpy": None}):
        with pytest.raises(ExplanationError, match="NumPy"):
            service._to_shap_mapping(np.array([0.1, 0.2]), ["f1", "f2"])
