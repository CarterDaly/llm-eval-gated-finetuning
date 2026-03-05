class ModelNotFoundError(Exception):
    """Raised when a requested model cannot be found."""


class ModelLoadError(Exception):
    """Raised when a model fails to load from storage."""


class InferenceError(Exception):
    """Raised when inference fails for a loaded model."""


class ExplanationError(Exception):
    """Raised when SHAP explanation cannot be computed."""
