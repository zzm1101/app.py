from app import app
from models.database import db
from models.influence_models import MlModelType

with app.app_context():
    # 检查是否已有数据
    if MlModelType.query.count() == 0:
        default_models = [
            ('lightgbm', 'LightGBM', '{"n_estimators":200, "learning_rate":0.05, "num_leaves":31}'),
            ('xgboost', 'XGBoost', '{"n_estimators":200, "learning_rate":0.05, "max_depth":5}'),
            ('random_forest', '随机森林', '{"n_estimators":200, "max_depth":10}'),
            ('decision_tree', '决策树', '{"max_depth":5}'),
            ('linear', '线性回归', '{}'),
        ]
        for key, name, params in default_models:
            db.session.add(MlModelType(model_key=key, display_name=name, default_params=params))
        db.session.commit()
        print("✅ 模型类型数据已初始化")
    else:
        print("模型类型数据已存在")