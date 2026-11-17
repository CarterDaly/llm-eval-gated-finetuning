from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Same alphabet as classical models: letters, digits, '.', '_', '-', no leading dot.
SAFE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class GenerationParams(BaseModel):
    """Decoding settings. The defaults are greedy, so eval runs are repeatable."""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    seed: Optional[int] = None


class LLMModelRequest(BaseModel):
    """JSON body for registering an LLM on POST /models."""
    kind: Literal["llm"]
    name: str = Field(pattern=SAFE_NAME_PATTERN, max_length=200)
    version: str = Field(pattern=SAFE_NAME_PATTERN, max_length=200)
    # Hugging Face repo id, e.g. "Qwen/Qwen2.5-1.5B-Instruct"
    base_model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
    # A full commit hash, not a branch: "main" can move, which silently changes
    # the weights and chat template behind an existing model version.
    base_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_uri: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    adapter_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mlflow_run_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    parent_model_id: Optional[str] = None
    generation_config: GenerationParams = Field(default_factory=GenerationParams)

    @model_validator(mode="after")
    def _adapter_needs_hash(self) -> "LLMModelRequest":
        if (self.adapter_uri is None) != (self.adapter_sha256 is None):
            raise ValueError("adapter_uri and adapter_sha256 must be given together")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=32_000)


class FunctionSpec(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$", max_length=64)
    description: Optional[str] = None
    # JSON Schema for the arguments
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """A tool the model may call, in the OpenAI format Qwen's chat template expects."""
    type: Literal["function"]
    function: FunctionSpec


class GenerationRequest(BaseModel):
    """JSON body for POST /generations/model/<name>/version/<version>."""
    messages: List[Message] = Field(min_length=1, max_length=50)
    tools: Optional[List[ToolSpec]] = Field(default=None, min_length=1, max_length=64)
    # Overrides the model's generation_config, field by field
    params: Optional[GenerationParams] = None
    # live: real traffic. eval: a golden/vault example. replay: the traffic simulator.
    source: Literal["live", "eval", "replay"] = "live"
    example_id: Optional[str] = Field(default=None, min_length=1, max_length=400)
    tags: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _dataset_traffic_needs_example_id(self) -> "GenerationRequest":
        # Paired candidate-vs-champion comparisons join generations on example_id
        if self.source != "live" and self.example_id is None:
            raise ValueError("example_id is required when source is 'eval' or 'replay'")
        return self
