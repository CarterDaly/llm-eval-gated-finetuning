class ModelLoadError(Exception):
    """Raised when a model fails to load from storage."""


class InferenceError(Exception):
    """Raised when inference fails for a loaded model."""


class ExplanationError(Exception):
    """Raised when SHAP explanation cannot be computed."""


class EmbeddingError(Exception):
    """Raised when UMAP embedding cannot be computed."""


class MlflowLogError(Exception):
    """Raised when MLflow logging fails."""


class ArizeLogError(Exception):
    """Raised when Arize model monitoring logging fails."""


class LLMError(Exception):
    """Raised when a local LLM (Ollama) API call fails."""
