# app.py
from flask import Flask
from config import Config
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config.from_object(Config)

from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()
csrf.init_app(app)  

from extensions import cache

cache.init_app(app)

from models.database import db

db.init_app(app)

from routes import register_blueprints

register_blueprints(app)

from models.models import CTQConfig, ProductionData, LossResult, DecayConfig, SPCRecord
from sqlalchemy import text
from utils import normalize_product_item

with app.app_context():
    db.create_all()
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

    # 注意：移除了全量的数据清洗 UPDATE，因为已经在数据导入时用 normalize_product_item 处理
    # 仅当数据库刚创建时插入默认 CTQ 配置
    if not CTQConfig.query.first():
        default_ctq = [
            CTQConfig(ctq_name="蛋白质含量", feature_type="nominal", process_link="原料标准化", is_ccp="是",
                      fmea_severity=6, gb_code="GB 19302-2010", target_m=3.1, usl=3.5, lsl=2.9, delta0=0.6,
                      delta=0.2, a0=3000, a=800, hidden_loss_coef=1.2, version=1),
            CTQConfig(ctq_name="滴定酸度", feature_type="nominal", process_link="发酵环节", is_ccp="是",
                      fmea_severity=5, gb_code="GB 19302-2010", target_m=75, usl=85, lsl=70, delta0=15,
                      delta=5, a0=2500, a=600, hidden_loss_coef=1.1, version=1),
            CTQConfig(ctq_name="灌装净含量", feature_type="nominal", process_link="灌装环节", is_ccp="否",
                      fmea_severity=4, gb_code="JJF 1070", target_m=200, usl=204.5, lsl=195.5, delta0=9,
                      delta=4.5, a0=1.2, a=0.3, asymmetric_loss="是", k_upper=0.0037, k_lower=0.0148,
                      a_upper=0.6, a_lower=1.2, hidden_loss_coef=1.0, version=1),
            CTQConfig(ctq_name="菌落总数", feature_type="smaller", process_link="成品检验", is_ccp="是",
                      fmea_severity=10, gb_code="GB 19302-2010", target_m=0, usl=100, lsl=0, delta0=100,
                      delta=50, a0=50000, a=50000, hidden_loss_coef=5.0, version=1),
            CTQConfig(ctq_name="乳清析出率", feature_type="smaller", process_link="发酵环节", is_ccp="否",
                      fmea_severity=6, gb_code="内控标准", target_m=0, usl=5, lsl=0, delta0=5,
                      delta=2, a0=2500, a=700, hidden_loss_coef=1.3, version=1),
            CTQConfig(ctq_name="保质期终点活菌数", feature_type="larger", process_link="仓储物流", is_ccp="是",
                      fmea_severity=8, gb_code="GB 19302-2010", target_m=1e7, usl=1e9, lsl=1e6, delta0=9e6,
                      delta=5e6, a0=3000, a=0, hidden_loss_coef=1.5, version=1),
            CTQConfig(ctq_name="冷链运输温度", feature_type="smaller", process_link="仓储物流", is_ccp="是",
                      fmea_severity=9, gb_code="GB 14881-2013", target_m=2, usl=6, lsl=0, delta0=4,
                      delta=2, a0=8000, a=1200, hidden_loss_coef=1.8, version=1),
        ]
        db.session.add_all(default_ctq)
        db.session.commit()
        print("✅ 已初始化默认CTQ配置")

    # 可选：缓存预热
    if app.config['CACHE_WARMUP']:
        try:
            from routes.dashboard import _build_dashboard_data

            _build_dashboard_data(LossResult.query, include_spc=False)
            print("✅ 缓存预热完成")
        except Exception as e:
            print(f"⚠️ 缓存预热失败: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("✅ 酸奶工厂田口QLF质量损失系统启动成功（性能优化版）")
    print("🌐 请打开浏览器访问：http://127.0.0.1:5000")
    if app.config.get('ENABLE_ML_INFLUENCE', False):
        print("🧠 CTQ影响因素分析模块已启用（机器学习）")
    else:
        print("🧠 CTQ影响因素分析模块未启用（如需使用请设置 ENABLE_ML_INFLUENCE=true）")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)