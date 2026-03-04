import logging

from flask import jsonify, request

from sqlalchemy.exc import IntegrityError

from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.user_model import UserSQL
from data5580_hw.models.user import User

from data5580_hw.models.prediction import Prediction, Model
from data5580_hw.services.model_service import model_service
from data5580_hw.services.database.prediction import PredictionSQL, ModelSQL



logger = logging.getLogger(__name__)


class PredictionController:

    @staticmethod
    def create_prediction(model_name: str, model_version: str) -> tuple[str, int]:
    
        prediction = Prediction.model_validate(request.get_json(force=True))

        prediction.model_name = model_name
        prediction.model_version = model_version

        # TODO: Load Model
        model = Model(name=model_name, version=model_version)
        prediction.model = model

        label = model_service.create_inference(Model, prediction = prediction)
        prediction.label = label

        prediction_sql = PredictionSQL.from_prediction(prediction, model)

        db.session.add(prediction_sql)

        db.session.commit()

        prediction_sql: PredictionSQL = db.session.query(PredictionSQL).filter(PredictionSQL.id == prediction_sql.id).first()

        prediction = prediction_sql.to_prediction()

        return prediction.model_dump_json(), 200

    @staticmethod
    def get_prediction_by_id(prediction_id: str) -> tuple[str, int]:
        pass
        # logger.debug(f'Got user_id {user_id}')

        # user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()

        # user = User.model_validate(user_sql, from_attributes=True)

        # if not user:
        #     return jsonify({}), 404

        # logging.info(f"user created, {user.id} for {user.email}")

        # return user.model_dump_json(), 200

    @staticmethod
    def update_actual(prediction_id: str) -> tuple[str, int]:
        pass

        # user = User.model_validate(request.get_json(force=True))

        # user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()

        # user_sql.name = user.name
        # user_sql.email = user.email

        # db.session.commit()

        # return user.model_dump_json(), 200

prediction_controller = PredictionController()