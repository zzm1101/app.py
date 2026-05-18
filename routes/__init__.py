# routes/__init__.py
from routes.dashboard import dashboard_bp
from routes.ctq_manage import ctq_bp
from routes.production_data import production_bp
from routes.k_value_calc import k_value_bp
from routes.target_calc import target_bp
from routes.loss_analysis import loss_bp
from routes.improvement import improvement_bp
from routes.help import help_bp
from routes.spc_page import spc_bp
from routes.influence import influence_bp
def register_blueprints(app):
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(ctq_bp)
    app.register_blueprint(production_bp)
    app.register_blueprint(k_value_bp)
    app.register_blueprint(target_bp)
    app.register_blueprint(loss_bp)
    app.register_blueprint(improvement_bp)
    app.register_blueprint(help_bp)
    app.register_blueprint(spc_bp)
    if app.config.get('ENABLE_ML_INFLUENCE', False):
        app.register_blueprint(influence_bp)