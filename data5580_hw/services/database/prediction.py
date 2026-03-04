from typing import List

from sqlalchemy import CLOB, ForeignKey
from sqlalchemy.orm import relationship, mapped_column, Mapped

from datetime import datetime

from data5580_hw.models.prediction import Prediction, Model
from data5580_hw.services.database.database_client import db


class ModelSQL(db.Model):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)

    model_type: Mapped[str] = mapped_column(db.String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(db.String(4000), nullable=False)
    model_version: Mapped[str] = mapped_column(db.String(4000), nullable=False)

    created: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)
    updated: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)

    
    predictions: Mapped[List['PredictionSQL']] = relationship(back_populates="model")


class PredictionSQL(db.Model):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(db.String(120), primary_key=True)

    features: Mapped[dict] = mapped_column(CLOB, nullable=False)
    tags: Mapped[dict] = mapped_column(CLOB)
    label: Mapped[str] = mapped_column(db.String(120), nullable=False)
    actual: Mapped[str] = mapped_column(db.String(120), nullable=True)
    threshold: Mapped[float] = mapped_column(db.Float, nullable=True)

    created: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)
    updated: Mapped[datetime] = mapped_column(db.DateTime, nullable=True)

    model_id: Mapped[str] = mapped_column(db.String(120), ForeignKey("models.id"), nullable=False)
    model: Mapped['ModelSQL'] = relationship(back_populates="predictions")

    @classmethod
    def from_prediction(cls, prediction: Prediction, model: Model) -> 'PredictionSQL':
        return cls(
            id=prediction.id,
            features=prediction.features,
            tags=prediction.tags,
            label=prediction.label,
            actual=prediction.actual,
            threshold=prediction.threshold,
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
            created=self.created,
            updated=self.updated,
            model=Model(
                id=self.model_id,
                name=self.model.model_name,
                version=self.model.model_version,
                type=self.model.model_type
            )
        )
    
    def __repr__(self) -> str:
        return f"<PredictionSQL id={self.id} model_id={self.model_id} label={self.label} actual={self.actual}>"