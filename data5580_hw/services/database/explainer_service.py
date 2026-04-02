from data5580_hw.services.database.database_client import db
from data5580_hw.gateways import mlflowgateway
from data5580_hw.services.database.prediction import PredictionSQL

class ExplainerService:

    @staticmethod
    def create_explanation(model: Model, prediction: Prediction) -> Explanations:
        explanations = ._

    return explanations

@celery.task(bind=True)
@staticmethod
def run_prediction_id(prediction_id: str) -> None:

    prediction_sql: PredictionSQL = db.session.query(PredictionSQL).filter(PredictionSQL.id == prediction_id).first()
    prediction = prediction_sql.to_prediction()

    model = prediction.model
    model: Model = mlflow.gateway.get_model(model.name, model.version)

    explanations = ExplainerService.create_explanation(model, prediction)

    prediction.explanations = explanations

    explanation_sql = ExplanationSql.from_prediction(prediction)

    db.session.add_all(explanation_sql)
    db.session.commit()

    return prediction_id


explainer_service = ExplainerService()