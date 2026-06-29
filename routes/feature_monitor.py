# routes/feature_monitor.py
import json
import numpy as np
from flask import Blueprint, render_template, request, jsonify
from models.database import db
from models.models import CTQConfig, ProductionData
from models.influence_models import CtqFeatureValue
from models.feature_spec_limit import FeatureSpecLimit
from services.spc_service import compute_individual_control_chart
from sqlalchemy import or_

feature_monitor_bp = Blueprint('feature_monitor', __name__, url_prefix='/feature-monitor')


@feature_monitor_bp.route('/')
def index():
    """影响因素控制图页面入口"""
    items = db.session.query(ProductionData.product_item).distinct() \
        .filter(ProductionData.product_item.isnot(None)).all()
    product_items = sorted([i[0] for i in items if i[0]])
    return render_template('feature_spc.html',
                           product_items=product_items,
                           active_page='feature_monitor')


@feature_monitor_bp.route('/api/batch_data')
def batch_feature_data():
    """获取指定品项下所有 CTQ 关联的特征控制图数据"""
    product_item = request.args.get('product_item', '').strip()
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    rules = request.args.get('rules', '')

    active_rules = []
    if rules:
        try:
            active_rules = [int(x.strip()) for x in rules.split(',') if x.strip()]
        except:
            active_rules = list(range(1, 9))
    else:
        active_rules = list(range(1, 9))

    # 获取该品项下所有启用的 CTQ（包括通用配置）
    ctq_list = CTQConfig.query.filter(
        CTQConfig.status == '启用',
        or_(
            CTQConfig.product_item == product_item,
            CTQConfig.product_item.is_(None)
        )
    ).all()

    result = {
        'product_item': product_item or '所有品项',
        'ctqs': []
    }

    for ctq in ctq_list:
        # 获取该 CTQ 下所有有数据的特征名称
        batch_subq = db.session.query(ProductionData.batch_no).filter(
            ProductionData.ctq_id == ctq.ctq_id
        )
        if product_item:
            batch_subq = batch_subq.filter(ProductionData.product_item == product_item)
        batch_subq = batch_subq.subquery()

        features = db.session.query(CtqFeatureValue.feature_name).filter(
            CtqFeatureValue.ctq_id == ctq.ctq_id,
            CtqFeatureValue.batch_no.in_(batch_subq)
        ).distinct().all()
        feature_names = [f[0] for f in features]

        if not feature_names:
            continue

        ctq_data = {
            'ctq_id': ctq.ctq_id,
            'ctq_name': ctq.ctq_name,
            'features': []
        }

        for feat_name in feature_names:
            query = db.session.query(
                ProductionData.batch_no,
                ProductionData.produce_date,
                CtqFeatureValue.feature_value
            ).join(
                CtqFeatureValue,
                (ProductionData.batch_no == CtqFeatureValue.batch_no) &
                (ProductionData.ctq_id == CtqFeatureValue.ctq_id)
            ).filter(
                CtqFeatureValue.ctq_id == ctq.ctq_id,
                CtqFeatureValue.feature_name == feat_name,
                CtqFeatureValue.feature_value.isnot(None)
            )
            if product_item:
                query = query.filter(ProductionData.product_item == product_item)
            if start_date:
                query = query.filter(ProductionData.produce_date >= start_date)
            if end_date:
                query = query.filter(ProductionData.produce_date <= end_date)

            results = query.order_by(ProductionData.produce_date).all()
            if len(results) < 2:
                continue

            values = [float(r.feature_value) for r in results]
            batch_labels = [r.batch_no for r in results]
            dates = [r.produce_date.strftime('%Y-%m-%d') for r in results]

            chart_data = compute_individual_control_chart(values, active_rules)
            if chart_data.get('error'):
                continue

            chart_data['labels'] = batch_labels
            chart_data['dates'] = dates
            chart_data['feature_name'] = feat_name
            chart_data['ctq_id'] = ctq.ctq_id  # 用于前端关联界限配置

            ctq_data['features'].append(chart_data)

        if ctq_data['features']:
            result['ctqs'].append(ctq_data)

    return jsonify(result)


# ================== 规格限与固定控制限配置 API ==================

@feature_monitor_bp.route('/api/feature_limits', methods=['GET'])
def get_feature_limits():
    """获取指定品项下所有特征的界限配置"""
    product_item = request.args.get('product_item', '')
    if not product_item:
        return jsonify({'limits': {}})

    records = FeatureSpecLimit.query.filter_by(product_item=product_item).all()
    limits = {}
    for r in records:
        key = f"{r.ctq_id}_{r.feature_name}"
        limits[key] = {
            'usl': r.usl,
            'lsl': r.lsl,
            'ucl': r.ucl,
            'lcl': r.lcl,
            'cl': r.cl,
            'show_spec': r.show_spec,
            'use_fixed_control': r.use_fixed_control
        }
    return jsonify({'limits': limits})


@feature_monitor_bp.route('/api/feature_limits', methods=['POST'])
def save_feature_limits():
    """批量保存界限配置"""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': '无效的请求数据'}), 400

    product_item = data.get('product_item')
    if not product_item:
        return jsonify({'status': 'error', 'message': '缺少品项信息'}), 400

    limits_data = data.get('limits', [])
    try:
        for item in limits_data:
            rec = FeatureSpecLimit.query.filter_by(
                product_item=product_item,
                ctq_id=item['ctq_id'],
                feature_name=item['feature_name']
            ).first()
            if not rec:
                rec = FeatureSpecLimit(
                    product_item=product_item,
                    ctq_id=item['ctq_id'],
                    feature_name=item['feature_name']
                )
                db.session.add(rec)

            rec.usl = item.get('usl')
            rec.lsl = item.get('lsl')
            rec.ucl = item.get('ucl')
            rec.lcl = item.get('lcl')
            rec.cl = item.get('cl')
            rec.show_spec = item.get('show_spec', False)
            rec.use_fixed_control = item.get('use_fixed_control', False)

        db.session.commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500