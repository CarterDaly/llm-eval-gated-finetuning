from data5580_hw.models.prediction import Model, Prediction, UmapConfig
from data5580_hw.services.errors import EmbeddingError
from data5580_hw.services.umap_service import UmapService


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
