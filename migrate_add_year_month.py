#!/usr/bin/env python
"""一次性迁移脚本：添加 production_year_month 字段并回填历史数据"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from app import app
from models.database import db
from models.models import ProductionData, LossResult

def run_migration():
    with app.app_context():
        # 1. 添加字段（使用 text() 包裹原生 SQL）
        try:
            db.session.execute(text('ALTER TABLE production_data ADD COLUMN production_year_month VARCHAR(7)'))
            print("✅ production_data 表添加字段 production_year_month 成功")
        except Exception as e:
            if "duplicate column name" in str(e).lower():
                print("⚠️ production_year_month 字段已存在，跳过添加")
            else:
                print(f"添加字段时出错: {e}")
                db.session.rollback()

        try:
            db.session.execute(text('ALTER TABLE loss_result ADD COLUMN production_year_month VARCHAR(7)'))
            print("✅ loss_result 表添加字段 production_year_month 成功")
        except Exception as e:
            if "duplicate column name" in str(e).lower():
                print("⚠️ production_year_month 字段已存在，跳过添加")
            else:
                print(f"添加字段时出错: {e}")
                db.session.rollback()

        db.session.commit()

        # 2. 回填 ProductionData
        count = 0
        for record in ProductionData.query.filter(ProductionData.production_year_month.is_(None)).all():
            if record.produce_date:
                record.production_year_month = record.produce_date.strftime('%Y-%m')
                db.session.add(record)
                count += 1
                if count % 1000 == 0:
                    db.session.commit()
                    print(f"已处理 ProductionData {count} 条")
        db.session.commit()
        print(f"✅ ProductionData 回填完成，共 {count} 条")

        # 3. 回填 LossResult
        count = 0
        for record in LossResult.query.filter(LossResult.production_year_month.is_(None)).all():
            if record.produce_date:
                record.production_year_month = record.produce_date.strftime('%Y-%m')
                db.session.add(record)
                count += 1
                if count % 1000 == 0:
                    db.session.commit()
                    print(f"已处理 LossResult {count} 条")
        db.session.commit()
        print(f"✅ LossResult 回填完成，共 {count} 条")

        print("🎉 数据迁移全部完成！")

if __name__ == '__main__':
    run_migration()