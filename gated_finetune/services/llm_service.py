import logging
import os

import numpy as np
import requests

from gated_finetune.services.errors import LLMError

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_MODEL = "qwen2.5:14b"
_OLLAMA_TIMEOUT = 60
_OLLAMA_MAX_TOKENS = 600
_DEFAULT_NEAREST_N = 5


class LLMService:

    def find_nearest_predictions(self, target_prediction, all_predictions, n: int = _DEFAULT_NEAREST_N) -> list:
        """Return the n most similar predictions using cosine similarity on embeddings."""
        target_embeddings = target_prediction.embeddings
        if not target_embeddings:
            return []

        axis_keys = list(target_embeddings.keys())
        target_vec = np.array([target_embeddings[k] for k in axis_keys], dtype=float)
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return []

        scored = []
        for pred in all_predictions:
            if pred.id == target_prediction.id:
                continue
            if not pred.embeddings:
                continue
            vec = np.array([pred.embeddings.get(k, 0.0) for k in axis_keys], dtype=float)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            similarity = float(np.dot(target_vec, vec) / (target_norm * vec_norm))
            scored.append((similarity, pred))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pred for _, pred in scored[:n]]

    def generate_summary(self, prediction, nearest_predictions: list) -> str:
        """Call the local Ollama/Qwen2.5 LLM to produce a plain-English prediction explanation."""
        base_url = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_BASE_URL)
        model = os.environ.get("OLLAMA_MODEL", _OLLAMA_MODEL)

        try:
            prompt = self._build_prompt(prediction, nearest_predictions)
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that explains machine learning "
                            "model predictions in plain, non-technical language."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"num_predict": _OLLAMA_MAX_TOKENS},
            }
            response = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=_OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("ollama_call_failed")
            raise LLMError(f"LLM API call failed: {exc}") from exc

    def _build_prompt(self, prediction, nearest_predictions: list) -> str:
        label = prediction.label
        features = prediction.features

        lines = [
            "A machine learning model produced the following prediction:",
            f"  - Predicted label: {label}",
            f"  - Input features: {features}",
        ]

        if nearest_predictions:
            lines.append(
                f"\nThe {len(nearest_predictions)} most similar historical predictions are:"
            )
            for i, near in enumerate(nearest_predictions, 1):
                lines.append(f"  {i}. Label: {near.label}, Features: {near.features}")
            lines.append("\nPlease provide a concise summary that includes:")
            lines.append("1. A plain-language explanation of what this prediction means.")
            lines.append(
                "2. Key differences between this prediction and the nearest similar ones "
                "(e.g., differences in predicted label, notable feature variances, "
                "or significant semantic changes)."
            )
        else:
            lines.append(
                "\nNo similar historical predictions were found for comparison."
            )
            lines.append(
                "\nPlease provide a plain-language explanation of what this prediction means "
                "and which features drove it."
            )

        lines.append("\nKeep the response concise and suitable for a non-technical audience.")
        return "\n".join(lines)


llm_service = LLMService()
