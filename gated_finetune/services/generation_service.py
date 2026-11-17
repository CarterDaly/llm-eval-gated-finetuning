"""Text generation for registered LLMs, through a pluggable serving backend.

LLM_BACKEND picks the backend for the whole deployment:
  - vllm (default): an OpenAI-compatible vLLM server at VLLM_BASE_URL, on the GPU box.
    The only backend whose outputs count for gating.
  - transformers: local HF transformers + peft, for development on the Mac.
"""
import os
from dataclasses import dataclass
from typing import Optional

import requests

from gated_finetune.services.database.prediction import ModelSQL
from gated_finetune.services.errors import GenerationBackendError

_DEFAULT_VLLM_BASE_URL = "http://localhost:8000"
_DEFAULT_VLLM_TIMEOUT = 60


@dataclass(frozen=True)
class GenerationResult:
    output_text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    served_model: str


class VLLMBackend:
    """Calls vLLM's OpenAI-compatible /v1/chat/completions endpoint.

    vLLM applies the base model's own chat template (from the pinned revision it
    was started with), the same template TRL uses for training. Do not start
    vLLM with --chat-template, or serving and training prompts would differ.

    Naming convention for the served model:
      - base model only: the Hugging Face id it was started with (`vllm serve <base_model>`)
      - base + adapter: the adapter is loaded into vLLM under the model id
        (e.g. "model-fc-bot-v2"), via --lora-modules or /v1/load_lora_adapter.
    """

    name = "vllm"

    def generate(self, model: ModelSQL, messages: list, tools: Optional[list], params: dict) -> GenerationResult:
        served_model = model.id if model.adapter_uri else model.base_model
        payload = {
            "model": served_model,
            "messages": messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "max_tokens": params["max_tokens"],
        }
        if params.get("seed") is not None:
            payload["seed"] = params["seed"]
        if tools:
            # Put the tools in the prompt but don't let vLLM parse the reply:
            # with tool_choice "none" the raw text (including <tool_call> tags)
            # comes back in `content`, and our own code checks parse it, so
            # scoring is identical whichever backend produced the output.
            payload["tools"] = tools
            payload["tool_choice"] = "none"

        base_url = os.environ.get("VLLM_BASE_URL", _DEFAULT_VLLM_BASE_URL).rstrip("/")
        headers = {}
        api_key = os.environ.get("VLLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = float(os.environ.get("VLLM_TIMEOUT", _DEFAULT_VLLM_TIMEOUT))

        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise GenerationBackendError(f"vLLM request failed: {exc.__class__.__name__}") from exc
        if response.status_code != 200:
            raise GenerationBackendError(f"vLLM returned HTTP {response.status_code}: {response.text[:500]}")

        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return GenerationResult(
                output_text=choice["message"]["content"] or "",
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                served_model=data.get("model", served_model),
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GenerationBackendError("vLLM returned an unexpected response body") from exc


class GenerationService:

    def __init__(self) -> None:
        self._backends = {VLLMBackend.name: VLLMBackend()}

    def backend_name(self) -> str:
        return os.environ.get("LLM_BACKEND", VLLMBackend.name).strip().lower()

    def generate(self, model: ModelSQL, messages: list, tools: Optional[list], params: dict) -> GenerationResult:
        name = self.backend_name()
        backend = self._backends.get(name)
        if backend is None:
            raise GenerationBackendError(f"Unknown LLM_BACKEND '{name}'; expected one of {sorted(self._backends)}")
        return backend.generate(model, messages, tools, params)


generation_service = GenerationService()
