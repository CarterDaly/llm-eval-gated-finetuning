
# Standardlibrary
import uuid

# Installed
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

# Local


def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db = SQLAlchemy(app)

    class User(db.Model):
        __tablename__ = "users"
        id = db.Column(db.String, primary_key=True)
        name = db.Column(db.String)
        email = db.Column(db.String)


    with app.app_context():
        db.drop_all()
        db.create_all()


    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Hello, Flask!"})

    @app.route("/add", methods=["POST"])
    def add():
        data = request.get_json()
        a = data.get("a")
        b = data.get("b")

        if a is None or b is None:
            return jsonify({"error": "Missing values"}), 400

        return jsonify({"result": a + b})

    @app.route("/user/{string: user_id}", methods=["GET"])
    def get_user_by_id(user_id: str):

        User.query.filter_by(id=user_id).first()

    @app.route("/user", methods=["POST"])
    def create_user():
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        id = uuid.uuid4().hex

        user = User(name=name, email=email, id=id)

        db.session.add(user)
        db.session.commit()

        return jsonify({"name": name, "email": email, "id": id}), 200

    return app




if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)