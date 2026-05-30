# app.py
from flask import Flask
from config import Config
import os
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config.from_object(Config)

from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()
csrf.init_app(app)

# ========== 豁免通用预测模块的 CSRF 保护（不影响原有业务） ==========
from routes.ml_tool import ml_tool_bp
csrf.exempt(ml_tool_bp)

from extensions import cache
cache.init_app(app)

from models.database import db
db.init_app(app)

from routes import register_blueprints
register_blueprints(app)

# ========== 豁免模拟数据生成器的 CSRF 保护 ==========
from routes.simulate import simulate_bp
csrf.exempt(simulate_bp)

# ========== 密封监测系统集成 ==========
from seal_monitor import seal_bp
app.register_blueprint(seal_bp)
csrf.exempt(seal_bp)   # 密封系统前端未使用 CSRF token

from models.models import CTQConfig, ProductionData, LossResult, DecayConfig, SPCRecord
from sqlalchemy import text, inspect
from utils import normalize_product_item


def load_default_ctqs():
    """从 data/default_ctqs.json 加载默认 CTQ 配置"""
    path = Path(__file__).parent / 'data' / 'default_ctqs.json'
    if not path.exists():
        app.logger.warning("default_ctqs.json 不存在，无法加载默认 CTQ")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


with app.app_context():
    db.create_all()

    # ========== 自动迁移：添加 production_year_month 列（若不存在） ==========
    inspector = inspect(db.engine)
    columns_pd = [col['name'] for col in inspector.get_columns('production_data')]
    columns_lr = [col['name'] for col in inspector.get_columns('loss_result')]

    if 'production_year_month' not in columns_pd:
        db.session.execute(text('ALTER TABLE production_data ADD COLUMN production_year_month VARCHAR(7)'))
        print("✅ 自动添加列 production_data.production_year_month")
    if 'production_year_month' not in columns_lr:
        db.session.execute(text('ALTER TABLE loss_result ADD COLUMN production_year_month VARCHAR(7)'))
        print("✅ 自动添加列 loss_result.production_year_month")
    db.session.commit()

    # 回填历史数据（仅对已有记录且 production_year_month 为空的行）
    updated_pd = 0
    for record in ProductionData.query.filter(ProductionData.production_year_month.is_(None)).all():
        if record.produce_date:
            record.production_year_month = record.produce_date.strftime('%Y-%m')
            db.session.add(record)
            updated_pd += 1
            if updated_pd % 1000 == 0:
                db.session.commit()
    if updated_pd:
        db.session.commit()
        print(f"✅ 回填 ProductionData.production_year_month 完成，共 {updated_pd} 条")

    updated_lr = 0
    for record in LossResult.query.filter(LossResult.production_year_month.is_(None)).all():
        if record.produce_date:
            record.production_year_month = record.produce_date.strftime('%Y-%m')
            db.session.add(record)
            updated_lr += 1
            if updated_lr % 1000 == 0:
                db.session.commit()
    if updated_lr:
        db.session.commit()
        print(f"✅ 回填 LossResult.production_year_month 完成，共 {updated_lr} 条")

    # ========== 原有初始化逻辑 ==========
    # 从数据库加载 ML 影响因素开关状态（持久化）
    from models.influence_models import SystemSetting
    setting = SystemSetting.query.filter_by(key='ENABLE_ML_INFLUENCE').first()
    if setting:
        app.config['ENABLE_ML_INFLUENCE'] = setting.value.lower() == 'true'
    else:
        # 首次启动，使用配置文件中的默认值，并写入数据库
        default_enabled = app.config.get('ENABLE_ML_INFLUENCE', True)
        db.session.add(SystemSetting(key='ENABLE_ML_INFLUENCE', value=str(default_enabled).lower()))
        db.session.commit()

    # 创建所有必要的索引（使用 IF NOT EXISTS 避免重复）
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_pd_batch_ctq ON production_data (batch_no, ctq_id)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_lr_batch_ctq ON loss_result (batch_no, ctq_id)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_lr_week ON loss_result (production_week)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_lr_month ON loss_result (production_month)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_lr_product_item ON loss_result (product_item)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_lr_product_line ON loss_result (product_line)'))
    db.session.execute(
        text('CREATE INDEX IF NOT EXISTS idx_lr_item_ctq_time ON loss_result (product_item, ctq_id, calc_time)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_pd_product_item ON production_data (product_item)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_pd_ctq_item ON production_data (ctq_id, product_item)'))
    # 新增索引（提升查询性能）
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_prod_date ON production_data (produce_date)'))
    db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_lr_product_item_ctq ON loss_result (product_item, ctq_id)'))
    db.session.execute(
        text('CREATE INDEX IF NOT EXISTS idx_ctq_feature_severity ON ctq_config (feature_type, fmea_severity)'))

    # 初始化默认 CTQ 配置（仅当表为空时）—— 从 JSON 加载
    if not CTQConfig.query.first():
        default_ctq_data = load_default_ctqs()
        for item in default_ctq_data:
            db.session.add(CTQConfig(**item))
        db.session.commit()
        print("✅ 已初始化默认CTQ配置（从 JSON 加载）")

    # 可选：缓存预热
    if app.config['CACHE_WARMUP']:
        try:
            from routes.dashboard import _build_dashboard_data
            _build_dashboard_data(LossResult.query, include_spc=False)
            print("✅ 缓存预热完成")
        except Exception as e:
            print(f"⚠️ 缓存预热失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("✅ 酸奶工厂田口QLF质量损失系统启动成功（性能优化版 + 通用预测模块）")
    print("🌐 请打开浏览器访问：http://127.0.0.1:5000")
    if app.config.get('ENABLE_ML_INFLUENCE', False):
        print("🧠 CTQ影响因素分析模块已启用（机器学习）")
    else:
        print("🧠 CTQ影响因素分析模块未启用（如需使用请设置 ENABLE_ML_INFLUENCE=true）")
    print("📊 通用预测模块入口：http://127.0.0.1:5000/ml-tool")
    print("🔧 密封磨损监测系统入口：http://127.0.0.1:5000/seal")
    print("📈 模拟数据生成模块入口：http://127.0.0.1:5000/simulate")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5001)