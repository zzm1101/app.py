# models/models.py
from datetime import datetime
from .database import db

class CTQConfig(db.Model):
    __tablename__ = "ctq_config"
    ctq_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_item = db.Column(db.String(50), default=None)
    ctq_name = db.Column(db.String(50), nullable=False)
    feature_type = db.Column(db.String(20), nullable=False)
    process_link = db.Column(db.String(50))
    is_ccp = db.Column(db.String(5), default="否")
    fmea_severity = db.Column(db.Integer, default=5)
    gb_code = db.Column(db.String(50))
    target_m = db.Column(db.Float, default=0)
    usl = db.Column(db.Float, nullable=False)
    lsl = db.Column(db.Float, nullable=False)
    delta0 = db.Column(db.Float)
    delta = db.Column(db.Float)
    a0 = db.Column(db.Float)
    a = db.Column(db.Float)
    asymmetric_loss = db.Column(db.String(5), default="否")
    k_upper = db.Column(db.Float, default=0)
    k_lower = db.Column(db.Float, default=0)
    a_upper = db.Column(db.Float, default=0)
    a_lower = db.Column(db.Float, default=0)
    hidden_loss_coef = db.Column(db.Float, default=1.0)
    status = db.Column(db.String(10), default="启用")
    create_time = db.Column(db.DateTime, default=datetime.now)
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    version = db.Column(db.Integer, default=1)
    update_user = db.Column(db.String(50), nullable=True)
    __table_args__ = (
        db.Index('idx_ctq_item_name', 'product_item', 'ctq_name'),
        db.Index('idx_ctq_status', 'status'),
        db.Index('idx_product_item', 'product_item'),
        db.Index('idx_ctq_feature_severity', 'feature_type', 'fmea_severity'),  # 新增
    )

class ProductionData(db.Model):
    __tablename__ = "production_data"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    batch_no = db.Column(db.String(50), nullable=False)
    produce_date = db.Column(db.Date, nullable=False)
    product_line = db.Column(db.String(20))
    work_shift = db.Column(db.String(20))
    product_item = db.Column(db.String(50))
    sample_no = db.Column(db.String(50))
    ctq_id = db.Column(db.Integer, db.ForeignKey("ctq_config.ctq_id"), nullable=False)
    ctq_name = db.Column(db.String(50), nullable=False)
    measured_value = db.Column(db.Float, nullable=False)
    batch_quantity = db.Column(db.Integer)
    storage_days = db.Column(db.Integer, default=0)
    storage_temp = db.Column(db.Float, default=4)
    inspector = db.Column(db.String(20))
    production_week = db.Column(db.Integer)
    production_month = db.Column(db.Integer)
    production_year_month = db.Column(db.String(7))  # 新增：格式 "YYYY-MM"
    create_time = db.Column(db.DateTime, default=datetime.now)
    __table_args__ = (
        db.Index('idx_prod_batch_ctq', 'batch_no', 'ctq_id'),
        db.Index('idx_prod_ctq_item', 'ctq_id', 'product_item'),
        db.Index('idx_prod_date', 'produce_date'),               # 新增
        db.Index('idx_prod_batch_item', 'batch_no', 'product_item'),  # 新增
    )

class LossResult(db.Model):
    __tablename__ = "loss_result"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    batch_no = db.Column(db.String(50), nullable=False)
    produce_date = db.Column(db.Date, nullable=False)
    product_line = db.Column(db.String(20))
    product_item = db.Column(db.String(50))
    production_week = db.Column(db.Integer)
    production_month = db.Column(db.Integer)
    production_year_month = db.Column(db.String(7))  # 新增
    ctq_id = db.Column(db.Integer, db.ForeignKey("ctq_config.ctq_id"), nullable=False)
    ctq_name = db.Column(db.String(50), nullable=False)
    measured_value_mean = db.Column(db.Float)
    measured_value_std = db.Column(db.Float)
    sample_count = db.Column(db.Integer, default=1)
    target_deviation = db.Column(db.Float)
    unit_expected_loss = db.Column(db.Float)
    batch_quantity = db.Column(db.Integer)
    batch_total_loss = db.Column(db.Float)
    batch_total_loss_with_hidden = db.Column(db.Float)
    feature_type = db.Column(db.String(20))
    process_link = db.Column(db.String(50))
    is_ccp = db.Column(db.String(5))
    is_gb_compliant = db.Column(db.String(5))
    paf_category = db.Column(db.String(20))
    std_source = db.Column(db.String(20), default="batch")
    calc_time = db.Column(db.DateTime, default=datetime.now)
    ctq_version = db.Column(db.Integer)
    __table_args__ = (
        db.Index('idx_lr_batch_ctq', 'batch_no', 'ctq_id'),
        db.Index('idx_lr_calc_time', 'calc_time'),
        db.Index('idx_lr_product_item_ctq', 'product_item', 'ctq_id'),  # 新增
    )

class DecayConfig(db.Model):
    __tablename__ = "decay_config"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ctq_id = db.Column(db.Integer, db.ForeignKey("ctq_config.ctq_id"), nullable=False)
    ctq_name = db.Column(db.String(50), nullable=False)
    shelf_life_days = db.Column(db.Integer, default=21)
    std_cold_temp = db.Column(db.Float, default=4)
    actual_avg_temp = db.Column(db.Float, default=6)
    temp_fluctuation = db.Column(db.Float, default=2)
    temp_coef = db.Column(db.Float, default=0.12)
    process_std = db.Column(db.Float, default=500000)
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class SPCRecord(db.Model):
    __tablename__ = "spc_record"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ctq_id = db.Column(db.Integer, db.ForeignKey("ctq_config.ctq_id"), nullable=False)
    product_item = db.Column(db.String(50))
    chart_type = db.Column(db.String(20))
    analysis_time = db.Column(db.DateTime, default=datetime.now)
    result_json = db.Column(db.Text)
    create_time = db.Column(db.DateTime, default=datetime.now)