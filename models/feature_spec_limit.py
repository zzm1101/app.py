from models.database import db

class FeatureSpecLimit(db.Model):
    __tablename__ = 'feature_spec_limits'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_item = db.Column(db.String(100), nullable=False)
    ctq_id = db.Column(db.Integer, db.ForeignKey('ctq_config.ctq_id'), nullable=False)
    feature_name = db.Column(db.String(200), nullable=False)

    usl = db.Column(db.Float, nullable=True)
    lsl = db.Column(db.Float, nullable=True)
    ucl = db.Column(db.Float, nullable=True)
    lcl = db.Column(db.Float, nullable=True)
    cl  = db.Column(db.Float, nullable=True)

    show_spec = db.Column(db.Boolean, default=False)
    use_fixed_control = db.Column(db.Boolean, default=False)

    __table_args__ = (
        db.UniqueConstraint('product_item', 'ctq_id', 'feature_name', name='uq_feature_limit'),
    )