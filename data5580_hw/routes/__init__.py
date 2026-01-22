

def init_blueprints(app):

    from data5580_hw.routes.user import user
    app.register_blueprint(user)

    from data5580_hw.routes.metrics import metrics
    app.register_blueprint(metrics)