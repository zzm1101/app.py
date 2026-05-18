# routes/dashboard.py
from flask import Blueprint, render_template, jsonify, request
from models.models import LossResult, CTQConfig, db
from sqlalchemy import func, case
from services.spc_service import generate_spc_alerts
from extensions import cache
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/')


def _build_dashboard_data(base_query, include_spc=True, process_labels=None, process_vals=None):
    """优化后的聚合查询：一次查询获取多个指标"""
    agg_result = base_query.with_entities(
        func.sum(LossResult.batch_total_loss_with_hidden).label('total_loss'),
        func.count(func.distinct(LossResult.batch_no)).label('total_batches'),
        func.sum(case((LossResult.is_gb_compliant == "否", 1), else_=0)).label('non_compliant_batches')
    ).first()

    total_loss = agg_result.total_loss or 0.0
    total_batches = agg_result.total_batches or 0
    non_compliant_batches = agg_result.non_compliant_batches or 0
    compliant_batches = total_batches - non_compliant_batches
    compliance_rate = (compliant_batches / total_batches * 100) if total_batches > 0 else 100.0

    # CTQ损失分布
    ctq_loss_query = base_query.with_entities(
        LossResult.ctq_name,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).group_by(LossResult.ctq_name).all()
    ctq_loss_data = [{"name": name, "value": float(loss or 0)} for name, loss in ctq_loss_query if name]

    # 品项损失
    item_loss_query = base_query.with_entities(
        LossResult.product_item,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).group_by(LossResult.product_item).all()
    item_labels = [r[0] for r in item_loss_query if r[0]]
    item_values = [float(r[1] or 0) for r in item_loss_query if r[0]]

    # 产线损失
    line_loss_query = base_query.with_entities(
        LossResult.product_line,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).group_by(LossResult.product_line).all()
    line_labels = [r[0] for r in line_loss_query if r[0]]
    line_values = [float(r[1] or 0) for r in line_loss_query if r[0]]

    # 月度损失
    month_query = base_query.with_entities(
        LossResult.production_year_month,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).filter(LossResult.production_year_month != None) \
        .group_by(LossResult.production_year_month).order_by(LossResult.production_year_month).all()
    months = []
    month_losses = []
    for ym, loss in month_query:
        if ym:
            try:
                dt = datetime.strptime(ym, '%Y-%m')
                months.append(f"{dt.year}年{dt.month}月")
            except:
                months.append(ym)
            month_losses.append(float(loss or 0))

    # 环节损失：如果调用方未提供，则重新计算（但需要注意避免重复 join）
    if process_labels is None or process_vals is None:
        # 重新构建环节损失查询，避免使用可能已 join 的 base_query
        # 从 LossResult 开始，应用与 base_query 相同的筛选条件？无法获取，所以这里简单地从 LossResult 查询所有
        # 但为了与筛选同步，这里我们采用另一种方式：从 base_query 的语句中提取筛选条件？太复杂。
        # 简化：环节损失查询独立于筛选，但为了与页面筛选联动，我们需要在调用 index 时传入。
        # 因此，建议在 index 中计算并传入。
        # 临时方案：使用 base_query 的一个副本来重新 join，但使用 subquery 重置。
        # 由于 base_query 可能已经有 join，我们需要清除 join。使用 select_from(LossResult) 可能无效。
        # 最稳妥：在 index 函数中单独计算环节损失。
        pass

    data = {
        'total_loss': total_loss,
        'compliance_rate': compliance_rate,
        'ctq_loss_data': ctq_loss_data,
        'item_labels': item_labels,
        'item_values': item_values,
        'line_labels': line_labels,
        'line_values': line_values,
        'months': months,
        'month_losses': month_losses,
        'process_labels': process_labels if process_labels is not None else [],
        'process_vals': process_vals if process_vals is not None else [],
    }

    if include_spc:
        data['spc_alerts'] = generate_spc_alerts()
    else:
        data['spc_alerts'] = []

    return data


@dashboard_bp.route('/')
@cache.cached(timeout=60, key_prefix='dashboard_view', query_string=True)
def index():
    # 获取筛选参数
    ctq_filter = request.args.get('ctq', '')
    item_filter = request.args.get('item', '')
    line_filter = request.args.get('line', '')
    month_filter = request.args.get('month', type=int)
    process_filter = request.args.get('process', '')

    # 构建基础查询
    base_query = LossResult.query
    if ctq_filter:
        base_query = base_query.filter(LossResult.ctq_name == ctq_filter)
    if item_filter:
        base_query = base_query.filter(LossResult.product_item == item_filter)
    if line_filter:
        base_query = base_query.filter(LossResult.product_line == line_filter)
    if month_filter:
        base_query = base_query.filter(LossResult.production_month == month_filter)
    if process_filter:
        base_query = base_query.join(CTQConfig, LossResult.ctq_id == CTQConfig.ctq_id) \
            .filter(CTQConfig.process_link == process_filter)

    # 单独计算环节损失（避免重复 join）
    # 重新从 LossResult 开始，应用相同的筛选条件（除了 process_filter 需要特殊处理）
    process_query = LossResult.query
    if ctq_filter:
        process_query = process_query.filter(LossResult.ctq_name == ctq_filter)
    if item_filter:
        process_query = process_query.filter(LossResult.product_item == item_filter)
    if line_filter:
        process_query = process_query.filter(LossResult.product_line == line_filter)
    if month_filter:
        process_query = process_query.filter(LossResult.production_month == month_filter)
    # 环节筛选不能重复，且这里我们目的是按环节分组，所以不需要对 process_filter 进行过滤，而是要计算所有环节的损失
    # 但为了与 base_query 的筛选条件一致（例如用户筛选了品项），我们需要应用除 process_filter 外的所有条件
    # 然后进行 join 和分组
    process_query = process_query.join(CTQConfig, LossResult.ctq_id == CTQConfig.ctq_id) \
        .with_entities(CTQConfig.process_link, func.sum(LossResult.batch_total_loss).label('total_loss')) \
        .group_by(CTQConfig.process_link).all()
    process_labels = [r[0] for r in process_query if r[0]]
    process_vals = [float(r[1] or 0) for r in process_query if r[0]]

    dashboard_data = _build_dashboard_data(base_query, include_spc=True,
                                           process_labels=process_labels,
                                           process_vals=process_vals)

    ctq_count = CTQConfig.query.filter_by(status="启用").count()
    batch_count = db.session.query(func.count(func.distinct(LossResult.batch_no))).scalar() or 0

    # 高损失批次
    high_batches_query = base_query.with_entities(
        LossResult.batch_no,
        LossResult.product_item,
        func.sum(LossResult.batch_total_loss_with_hidden).label('total_loss')
    ).group_by(LossResult.batch_no, LossResult.product_item) \
        .having(func.sum(LossResult.batch_total_loss_with_hidden) > 10000) \
        .order_by(func.sum(LossResult.batch_total_loss_with_hidden).desc()).limit(20)
    high_batches = high_batches_query.all()

    return render_template(
        'dashboard.html',
        active_page='dashboard',
        total_loss=dashboard_data['total_loss'],
        compliance_rate=dashboard_data['compliance_rate'],
        ctq_loss_data=dashboard_data['ctq_loss_data'],
        item_labels=dashboard_data['item_labels'],
        item_values=dashboard_data['item_values'],
        line_labels=dashboard_data['line_labels'],
        line_values=dashboard_data['line_values'],
        months=dashboard_data['months'],
        month_losses=dashboard_data['month_losses'],
        process_labels=dashboard_data['process_labels'],
        process_vals=dashboard_data['process_vals'],
        spc_alerts=dashboard_data['spc_alerts'],
        ctq_count=ctq_count,
        batch_count=batch_count,
        high_batches=high_batches,
    )


@dashboard_bp.route('/api/dashboard_data')
@cache.cached(timeout=30, query_string=True)
def api_dashboard_data():
    ctq_filter = request.args.get('ctq', '')
    item_filter = request.args.get('item', '')
    line_filter = request.args.get('line', '')
    month_filter = request.args.get('month', type=int)
    process_filter = request.args.get('process', '')

    query = LossResult.query
    if ctq_filter:
        query = query.filter(LossResult.ctq_name == ctq_filter)
    if item_filter:
        query = query.filter(LossResult.product_item == item_filter)
    if line_filter:
        query = query.filter(LossResult.product_line == line_filter)
    if month_filter:
        query = query.filter(LossResult.production_month == month_filter)
    if process_filter:
        query = query.join(CTQConfig, LossResult.ctq_id == CTQConfig.ctq_id) \
            .filter(CTQConfig.process_link == process_filter)

    # 单独计算环节损失
    process_query = LossResult.query
    if ctq_filter:
        process_query = process_query.filter(LossResult.ctq_name == ctq_filter)
    if item_filter:
        process_query = process_query.filter(LossResult.product_item == item_filter)
    if line_filter:
        process_query = process_query.filter(LossResult.product_line == line_filter)
    if month_filter:
        process_query = process_query.filter(LossResult.production_month == month_filter)
    # 注意：不应用 process_filter 到 where 条件，因为我们要分组统计所有环节
    process_query = process_query.join(CTQConfig, LossResult.ctq_id == CTQConfig.ctq_id) \
        .with_entities(CTQConfig.process_link, func.sum(LossResult.batch_total_loss).label('total_loss')) \
        .group_by(CTQConfig.process_link).all()
    process_labels = [r[0] for r in process_query if r[0]]
    process_vals = [float(r[1] or 0) for r in process_query if r[0]]

    dashboard_data = _build_dashboard_data(query, include_spc=False,
                                           process_labels=process_labels,
                                           process_vals=process_vals)
    return jsonify(dashboard_data)