from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, mapped_column, Mapped

from datetime import datetime

from gated_finetune.models.prediction import Prediction, Model, Explanations, Explanation
from gated_finetune.services.database.database_client import db


class ExplanationSQL(db.Model):
    __tablename__ = "explanations"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)

    name: Mapped[str] = mapped_column(db.String(120), nullable=False)
    values: Mapped[dict] = mapped_column(db.JSON, nullable=False)

    prediction_id: Mapped[str] = mapped_column(db.String(120), ForeignKey("predictions.id"), nullable=False)
    prediction: Mapped['PredictionSQL'] = relationship(back_populates="explanations")

    @classmethod
    def from_explanation(cls, explanation: Explanation, prediction_id: str) -> "ExplanationSQL":
        return cls(
            id=explanation.id,
            name=explanation.name,
            values=explanation.values,
            prediction_id=prediction_id,
        )


class ModelSQL(db.Model):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)

    model_type: Mapped[str] = mapped_column(db.String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(db.String(4000), nullable=False)
    model_version: Mapped[str] = mapped_column(db.String(4000), nullable=False)
    artifact_path: Mapped[str] = mapped_column(db.String(4000), nullable=True)
    feature_names: Mapped[list] = mapped_column(db.JSON, nullable=True)
    mlflow_run_id: Mapped[str] = mapped_column(db.String(120), nullable=True)
    shap_artifact_path: Mapped[str] = mapped_column(db.String(4000), nullable=True)
    shap_logged_at: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)
    embedding_artifact_path: Mapped[str] = mapped_column(db.String(4000), nullable=True)
    embedding_logged_at: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)

    created: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)
    updated: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)

    predictions: Mapped[List['PredictionSQL']] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )


class PredictionSQL(db.Model):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)

    features: Mapped[dict] = mapped_column(db.JSON, nullable=False)
    tags: Mapped[dict] = mapped_column(db.JSON)
    label: Mapped[object] = mapped_column(db.JSON, nullable=False)
    actual: Mapped[object] = mapped_column(db.JSON, nullable=True)
    threshold: Mapped[float] = mapped_column(db.Float, nullable=True)
    embeddings: Mapped[dict] = mapped_column(db.JSON, nullable=True)

    created: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)
    updated: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)

    model_id: Mapped[str] = mapped_column(db.String(120), ForeignKey("models.id"), nullable=False)
    model: Mapped['ModelSQL'] = relationship(back_populates="predictions")
    explanations: Mapped[List['ExplanationSQL']] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
    )

    @classmethod
    def from_prediction(cls, prediction: Prediction, model: Model) -> 'PredictionSQL':
        label = prediction.label.item() if hasattr(prediction.label, "item") else prediction.label
        actual = prediction.actual.item() if hasattr(prediction.actual, "item") else prediction.actual
        return cls(
            id=prediction.id,
            features=prediction.features,
            tags=prediction.tags,
            label=label,
            actual=actual,
            threshold=prediction.threshold,
            embeddings=prediction.embeddings,
            created=prediction.created,
            updated=prediction.updated,
            model_id=model.id,
        )

    def to_prediction(self) -> Prediction:
        return Prediction(
            id=self.id,
            features=self.features,
            tags=self.tags,
            label=self.label,
            actual=self.actual,
            threshold=self.threshold,
            embeddings=self.embeddings,
            created=self.created,
            updated=self.updated,
            model=Model(
                id=self.model_id,
                name=self.model.model_name,
                version=self.model.model_version,
                type=self.model.model_type,
                artifact_path=self.model.artifact_path,
                feature_names=self.model.feature_names,
                mlflow_run_id=self.model.mlflow_run_id,
            ),
            explanations=Explanations.from_explanations(self.explanations)
        )

    def __repr__(self) -> str:
        return f"<PredictionSQL id={self.id} model_id={self.model_id} label={self.label} actual={self.actual}>"
