#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from models.database import db
from models.influence_models import MlModelType

def init():
    with app.app_context():
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
            print("✅ 模型类型数据初始化完成，共 {} 条".format(len(default_models)))
        else:
            print("模型类型数据已存在，无需初始化")

if __name__ == '__main__':
    init()