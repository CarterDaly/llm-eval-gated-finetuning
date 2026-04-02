"""Tests for ModelService – covers load_model, create_inference, and helpers."""
import pickle

import pytest

from data5580_hw.models.prediction import Model, Prediction
from data5580_hw.services.errors import InferenceError, ModelLoadError
from data5580_hw.services.model_service import ModelService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleModel:
    """Minimal sklearn-style model."""
    feature_names_in_ = ["a", "b"]

    def predict(self, X):
        return ["yes"]


class _LinearModel:
    coef_ = [0.1, 0.2]

    def predict(self, X):
        return [1.0]


class _TreeModel:
    tree_ = object()

    def predict(self, X):
        return [0]


class _EnsembleModel:
    estimators_ = []

    def predict(self, X):
        return [0]


class _NonIndexableOutput:
    """predict() returns something that raises on [0] access."""
    def predict(self, X):
        return 42  # int – output[0] raises TypeError


class _FailPredictModel:
    """predict() always raises."""
    def predict(self, X):
        raise RuntimeError("boom")


@pytest.fixture
def service():
    return ModelService()


def _make_model(tmp_path, obj=None, name="m", version="1"):
    obj = obj or _SimpleModel()
    p = tmp_path / "model.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return Model(name=name, version=version, artifact_path=str(p))


# ---------------------------------------------------------------------------
# load_model – cache hit
# ---------------------------------------------------------------------------

def test_load_model_cache_hit(service, tmp_path):
    model = _make_model(tmp_path)
    service.load_model(model)  # first load populates cache

    model2 = Model(name="m", version="1", artifact_path=str(tmp_path / "model.pkl"))
    service.load_model(model2)  # should hit cache
    assert model2._model is not None


def test_load_model_cache_hit_inherits_type_and_feature_names(service, tmp_path):
    model = _make_model(tmp_path)
    service.load_model(model)

    model2 = Model(name="m", version="1", artifact_path=str(tmp_path / "model.pkl"))
    service.load_model(model2)
    assert model2.feature_names == ["a", "b"]


# ---------------------------------------------------------------------------
# load_model – missing / bad path
# ---------------------------------------------------------------------------

def test_load_model_missing_artifact_path(service):
    model = Model(name="x", version="1")
    with pytest.raises(ModelLoadError, match="artifact path is missing"):
        service.load_model(model)


def test_load_model_file_not_found(service):
    model = Model(name="x", version="1", artifact_path="/nonexistent/model.pkl")
    with pytest.raises(ModelLoadError, match="artifact not found"):
        service.load_model(model)


def test_load_model_bad_pickle(service, tmp_path):
    bad = tmp_path / "model.pkl"
    bad.write_bytes(b"not a pickle")
    model = Model(name="x", version="1", artifact_path=str(bad))
    with pytest.raises(ModelLoadError, match="Failed to load"):
        service.load_model(model)


# ---------------------------------------------------------------------------
# create_inference
# ---------------------------------------------------------------------------

def test_create_inference_model_not_loaded(service):
    model = Model(name="x", version="1")
    prediction = Prediction(features={"a": 1})
    with pytest.raises(InferenceError, match="not loaded"):
        service.create_inference(model, prediction)


def test_create_inference_success(service, tmp_path):
    model = _make_model(tmp_path)
    service.load_model(model)
    prediction = Prediction(features={"a": 1, "b": 2})
    result = service.create_inference(model, prediction)
    assert result == "yes"


def test_create_inference_predict_raises(service, tmp_path):
    model = _make_model(tmp_path, obj=_FailPredictModel(), name="fail", version="1")
    service.load_model(model)
    prediction = Prediction(features={"a": 1})
    with pytest.raises(InferenceError, match="Failed to run inference"):
        service.create_inference(model, prediction)


def test_create_inference_non_indexable_output_fallback(service, tmp_path):
    model = _make_model(tmp_path, obj=_NonIndexableOutput(), name="nonidx", version="1")
    service.load_model(model)
    prediction = Prediction(features={"a": 1})
    result = service.create_inference(model, prediction)
    assert result == 42


# ---------------------------------------------------------------------------
# _infer_feature_names
# ---------------------------------------------------------------------------

def test_infer_feature_names_from_feature_names_in_(service):
    class M:
        feature_names_in_ = ["x", "y"]
    assert service._infer_feature_names(M()) == ["x", "y"]


def test_infer_feature_names_from_feature_names_(service):
    class M:
        feature_names_ = ["p", "q"]
    assert service._infer_feature_names(M()) == ["p", "q"]


def test_infer_feature_names_returns_none_when_absent(service):
    class M:
        pass
    assert service._infer_feature_names(M()) is None


# ---------------------------------------------------------------------------
# _infer_model_type
# ---------------------------------------------------------------------------

def test_infer_model_type_tree_via_tree_(service):
    assert service._infer_model_type(_TreeModel()) == "tree"


def test_infer_model_type_tree_via_estimators_(service):
    assert service._infer_model_type(_EnsembleModel()) == "tree"


def test_infer_model_type_linear_via_coef_(service):
    assert service._infer_model_type(_LinearModel()) == "linear"


def test_infer_model_type_tree_via_class_name_xgb(service):
    class XGBClassifier:
        pass
    assert service._infer_model_type(XGBClassifier()) == "tree"


def test_infer_model_type_tree_via_class_name_randomforest(service):
    class RandomForestClassifier:
        pass
    assert service._infer_model_type(RandomForestClassifier()) == "tree"


def test_infer_model_type_linear_via_class_name(service):
    class LogisticRegression:
        pass
    assert service._infer_model_type(LogisticRegression()) == "linear"


def test_infer_model_type_deep_via_class_name_torch(service):
    class TorchNet:
        pass
    assert service._infer_model_type(TorchNet()) == "deep"


def test_infer_model_type_generic_fallback(service):
    class UnknownModel:
        pass
    assert service._infer_model_type(UnknownModel()) == "generic"
