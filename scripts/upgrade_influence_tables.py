#!/usr/bin/env python
"""为影响因素模块增加类别特征支持字段及 training_params 字段"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from models.database import db
from sqlalchemy import text

def upgrade():
    with app.app_context():
        db.session.execute(text('PRAGMA foreign_keys=ON'))
        # 为 ctq_feature_values 增加字段
        for col in ['feature_type', 'categorical_mapping']:
            try:
                db.session.execute(text(f'ALTER TABLE ctq_feature_values ADD COLUMN {col} TEXT'))
                print(f"添加列 {col} 成功")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print(f"列 {col} 已存在")
                else:
                    print(f"添加列 {col} 失败: {e}")
        # 为 ml_model_metadata 增加字段（原有字段）
        for col in ['feature_types', 'feature_encodings', 'onehot_columns', 'feature_medians']:
            try:
                db.session.execute(text(f'ALTER TABLE ml_model_metadata ADD COLUMN {col} TEXT'))
                print(f"添加列 {col} 成功")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print(f"列 {col} 已存在")
                else:
                    print(f"添加列 {col} 失败: {e}")
        # 新增 training_params 列
        try:
            db.session.execute(text('ALTER TABLE ml_model_metadata ADD COLUMN training_params TEXT'))
            print("添加列 training_params 成功")
        except Exception as e:
            if "duplicate column name" in str(e).lower():
                print("列 training_params 已存在")
            else:
                print(f"添加列 training_params 失败: {e}")
        db.session.commit()
        print("升级完成")

if __name__ == '__main__':
    upgrade()