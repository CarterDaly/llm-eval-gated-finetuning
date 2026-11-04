import sys
from unittest.mock import patch

import pytest

from gated_finetune.models.prediction import Model, Prediction, UmapConfig
from gated_finetune.services.errors import EmbeddingError
from gated_finetune.services.umap_service import UmapService


def test_axis_names_handles_extra_components():
    service = UmapService()
    assert service._axis_names(2) == ["x", "y"]
    assert service._axis_names(4) == ["x", "y", "z", "w"]
    assert service._axis_names(6) == ["x", "y", "z", "w", "c1", "c2"]


def test_axis_names_requires_minimum_components():
    service = UmapService()
    try:
        service._axis_names(1)
    except EmbeddingError as exc:
        assert "n_components" in str(exc)
    else:
        raise AssertionError("Expected EmbeddingError for n_components < 2.")


def test_create_embedding_uses_umap(monkeypatch):
    service = UmapService()

    class DummyUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, _frame):
            return self

        def transform(self, _frame):
            return [[0.25, -0.5]]

    class DummyUmapModule:
        def UMAP(self, **kwargs):
            return DummyUMAP(**kwargs)

    monkeypatch.setattr(service, "_import_umap", lambda: DummyUmapModule())

    model = Model(name="demo", version="1", feature_names=["a", "b"])
    prediction = Prediction(features={"a": 1, "b": 2})
    config = UmapConfig(n_components=2, n_neighbors=10, min_dist=0.2)

    embedding = service.create_embedding(model, prediction, config)
    assert embedding["kind"] == "inputs"
    assert embedding["x"] == 0.25
    assert embedding["y"] == -0.5


def test_create_embedding_transform_raises(monkeypatch):
    service = UmapService()

    class _FailUMAP:
        def __init__(self, **kwargs): pass
        def fit(self, frame): return self
        def transform(self, frame): raise RuntimeError("transform failed")

    class _FailUmapModule:
        def UMAP(self, **kwargs): return _FailUMAP(**kwargs)

    monkeypatch.setattr(service, "_import_umap", lambda: _FailUmapModule())

    model = Model(name="fail", version="1", feature_names=["a", "b"])
    prediction = Prediction(features={"a": 1, "b": 2})
    config = UmapConfig(n_components=2)

    with pytest.raises(EmbeddingError, match="Failed to compute UMAP embedding"):
        service.create_embedding(model, prediction, config)


def test_create_embedding_cache_hit(monkeypatch):
    service = UmapService()

    class _CachedUMAP:
        call_count = 0

        def transform(self, frame):
            _CachedUMAP.call_count += 1
            return [[0.1, 0.2]]

    model = Model(name="cached", version="1", feature_names=["a", "b"])
    model._umap = _CachedUMAP()

    prediction = Prediction(features={"a": 1, "b": 2})
    config = UmapConfig(n_components=2)

    embedding = service.create_embedding(model, prediction, config)
    assert embedding["x"] == pytest.approx(0.1)
    assert _CachedUMAP.call_count == 1


def test_get_or_create_umap_fit_raises(monkeypatch):
    service = UmapService()

    class _BoomUMAP:
        def __init__(self, **kwargs): pass
        def fit(self, frame): raise RuntimeError("fit failed")

    class _BoomModule:
        def UMAP(self, **kwargs): return _BoomUMAP(**kwargs)

    monkeypatch.setattr(service, "_import_umap", lambda: _BoomModule())

    model = Model(name="fitfail", version="1", feature_names=["a", "b"])
    prediction = Prediction(features={"a": 1, "b": 2})
    config = UmapConfig(n_components=2)

    with pytest.raises(EmbeddingError, match="Failed to fit UMAP"):
        service.create_embedding(model, prediction, config)


def test_import_umap_not_installed():
    service = UmapService()
    blocked = {k: None for k in list(sys.modules.keys()) if k == "umap" or k.startswith("umap.")}
    blocked.setdefault("umap", None)
    with patch.dict(sys.modules, blocked):
        with pytest.raises(EmbeddingError, match="not installed"):
            service._import_umap()
