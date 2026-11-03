

def init_blueprints(app):

    from gated_finetune.routes.user import user
    app.register_blueprint(user)

    from gated_finetune.routes.prediction import prediction
    app.register_blueprint(prediction)

    from gated_finetune.routes.metrics import metrics
    app.register_blueprint(metrics)

    from gated_finetune.routes.task import task
    app.register_blueprint(task)

    from gated_finetune.routes.model import model
    app.register_blueprint(model)
