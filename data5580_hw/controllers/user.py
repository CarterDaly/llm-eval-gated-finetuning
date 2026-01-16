import logging

from flask import jsonify

from data5580_hw.services.databases.database_client import db
from data5580_hw.services.databases.user_model import UserSQL
from data5580_hw.models.user import User

logger = logging.getLogger(__name__)

class UserController:
    
    @staticmethod
    def create_user():



    @staticmethod
    def get_user_by_id(user_id: str) -> tuple[str, int]:
        
        logger.debug(f"Fetching user with ID: {user_id}")

        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()

        user = User.model_validate(user_sql, from_attributes=True)

        if not user:
            return jsonify({"error": "User not found"}), 404
        
        logging.info(f"User fetched {user.id} for {user.email}")

        return user.model_dump_json(), 200
    

user_controller = UserController()