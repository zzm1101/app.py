# models/influence_models.py
from datetime import datetime
from .database import db

class CtqFeatureValue(db.Model):
    __tablename__ = 'ctq_feature_values'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    batch_no = db.Column(db.String(50), nullable=False)
    ctq_id = db.Column(db.Integer, db.ForeignKey('ctq_config.ctq_id', ondelete='CASCADE'), nullable=False)
    feature_name = db.Column(db.String(100), nullable=False)
    feature_value = db.Column(db.Float)
    raw_value = db.Column(db.String(200))
    feature_type = db.Column(db.String(20), default='numeric')
    categorical_mapping = db.Column(db.Text)
    record_time = db.Column(db.DateTime, default=datetime.now)

    __table_args__ = (
        db.Index('idx_cfv_batch_ctq', 'batch_no', 'ctq_id'),
        db.Index('idx_cfv_name', 'feature_name'),
        db.UniqueConstraint('batch_no', 'ctq_id', 'feature_name', name='uq_cfv_batch_ctq_feature'),
    )

class CtqTargetOverride(db.Model):
    __tablename__ = 'ctq_target_overrides'
    batch_no = db.Column(db.String(50), primary_key=True)
    ctq_id = db.Column(db.Integer, db.ForeignKey('ctq_config.ctq_id', ondelete='CASCADE'), primary_key=True)
    target_value = db.Column(db.Float, nullable=False)
    source_file = db.Column(db.String(200))
    update_time = db.Column(db.DateTime, default=datetime.now)

class MlModelType(db.Model):
    __tablename__ = 'ml_model_types'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_key = db.Column(db.String(50), unique=True, nullable=False)
    display_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    default_params = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)

class MlModelMetadata(db.Model):
    __tablename__ = 'ml_model_metadata'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ctq_id = db.Column(db.Integer, db.ForeignKey('ctq_config.ctq_id', ondelete='CASCADE'), nullable=False)
    product_item = db.Column(db.String(50))
    model_type = db.Column(db.String(50), nullable=False)
    training_date = db.Column(db.DateTime, default=datetime.now)
    training_samples = db.Column(db.Integer)
    model_path = db.Column(db.String(255))
    feature_list = db.Column(db.Text)
    importance_json = db.Column(db.Text)
    hyperparams = db.Column(db.Text)
    r2_score = db.Column(db.Float)
    rmse = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=False)
    feature_types = db.Column(db.Text)
    feature_encodings = db.Column(db.Text)
    onehot_columns = db.Column(db.Text)
    feature_medians = db.Column(db.Text)
    training_params = db.Column(db.Text)   # 新增：存储训练参数配置（缺失值填充、标准化等）

    __table_args__ = (
        db.Index('idx_mlm_ctq', 'ctq_id', 'product_item', 'model_type'),
    )