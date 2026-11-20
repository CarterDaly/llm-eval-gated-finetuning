"""Embeds the user's part of an LLM request, for drift monitoring and clustering.

The model is fixed in code rather than configurable: embeddings from different
models aren't comparable, so changing it invalidates every stored vector and any
threshold tuned on them. Change it deliberately, with a DECISIONS.md entry.

fastembed runs the model with ONNX Runtime on the CPU (no torch, no GPU). It
downloads the weights on first use into FASTEMBED_CACHE_PATH (default: the
system temp directory).
"""
import threading
from importlib.metadata import version

from gated_finetune.services.errors import PromptEmbeddingError

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def prompt_text(messages: list) -> str:
    """The text that gets embedded: the user messages, in order.

    System prompts are left out because they are the same across requests and
    would pull every vector toward the same point. Assistant and tool turns are
    left out because they are model outputs or tool results, not user input.
    """
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


class PromptEmbeddingService:

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """Stored next to each vector: the model and the fastembed version that
        decides which copy of its weights was downloaded."""
        return f"{MODEL_NAME} (fastembed {version('fastembed')})"

    def _load(self):
        # Loaded once per process, on first use, so importing this module (the API,
        # tests) never triggers a download.
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=MODEL_NAME)
        return self._model

    def embed(self, messages: list) -> list[float]:
        text = prompt_text(messages)
        if not text.strip():
            raise PromptEmbeddingError("Request has no user message to embed")
        try:
            vector = next(iter(self._load().embed([text])))
        except Exception as exc:
            raise PromptEmbeddingError("Failed to compute prompt embedding") from exc
        return [float(x) for x in vector]


prompt_embedding_service = PromptEmbeddingService()
