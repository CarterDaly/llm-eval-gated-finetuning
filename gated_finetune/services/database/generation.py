from datetime import datetime

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gated_finetune.services.database.database_client import db


class GenerationSQL(db.Model):
    """One LLM request and its output.

    Kept separate from `predictions`, whose rows require tabular features and a
    label and carry SHAP/UMAP data that doesn't apply to text generation.
    """
    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)
    model_id: Mapped[str] = mapped_column(db.String(120), ForeignKey("models.id"), nullable=False, index=True)

    # Request. messages and tools are stored as sent; the prompt the model saw is
    # fully determined by them plus the base model's pinned chat template.
    messages: Mapped[list] = mapped_column(db.JSON, nullable=False)
    tools: Mapped[list] = mapped_column(db.JSON, nullable=True)
    params: Mapped[dict] = mapped_column(db.JSON, nullable=False)

    # Output. output_text is the raw completion; tool calls are parsed from it
    # later by the code checks, so scoring never depends on a backend's parser.
    output_text: Mapped[str] = mapped_column(db.Text, nullable=True)
    finish_reason: Mapped[str] = mapped_column(db.String(40), nullable=True)
    error: Mapped[str] = mapped_column(db.Text, nullable=True)

    # Serving
    backend: Mapped[str] = mapped_column(db.String(40), nullable=False)
    served_model: Mapped[str] = mapped_column(db.String(400), nullable=True)

    # Cost and speed. cost_usd is an estimate: tokens x cost_per_1k_tokens, with
    # the rate stored next to it so old rows stay interpretable if it changes.
    prompt_tokens: Mapped[int] = mapped_column(db.Integer, nullable=True)
    completion_tokens: Mapped[int] = mapped_column(db.Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(db.Float, nullable=True)
    cost_per_1k_tokens: Mapped[float] = mapped_column(db.Float, nullable=True)
    cost_usd: Mapped[float] = mapped_column(db.Float, nullable=True)

    # Filled in asynchronously after the response is sent
    prompt_embedding: Mapped[list] = mapped_column(db.JSON, nullable=True)
    embedding_model: Mapped[str] = mapped_column(db.String(400), nullable=True)

    # Context. source is live / eval / replay; example_id links eval and replay
    # requests to their dataset example, which paired comparisons join on.
    source: Mapped[str] = mapped_column(db.String(20), nullable=False, server_default="live")
    example_id: Mapped[str] = mapped_column(db.String(400), nullable=True, index=True)
    tags: Mapped[dict] = mapped_column(db.JSON, nullable=False)

    created: Mapped[datetime] = mapped_column(db.DateTime, nullable=False)

    model: Mapped["ModelSQL"] = relationship(back_populates="generations")  # noqa: F821
