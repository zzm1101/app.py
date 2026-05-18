# routes/loss_analysis.py
# 质量损失分析模块
from flask import Blueprint, render_template, request, jsonify, flash, send_file, redirect, url_for
from models.models import LossResult, CTQConfig, ProductionData, db
from services.taguchi_qlf import TaguchiQLFCore
from services.excel_service import export_excel
from services.spc_service import generate_spc_alerts
from sqlalchemy import func, and_
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from extensions import cache, clear_all_caches
from utils import normalize_product_item, to_float_or_zero, safe_division

loss_bp = Blueprint('loss', __name__, url_prefix='/loss')
FETCH_BATCH_SIZE = 1000


def build_ctq_map():
    ctq_map = {}
    for ctq in CTQConfig.query.filter_by(status="启用").all():
        item_key = ctq.product_item
        ctq_map[(item_key, ctq.ctq_name)] = ctq
        if item_key is None:
            ctq_map[(None, ctq.ctq_name)] = ctq
    return ctq_map


def find_ctq_config(product_item, ctq_name, ctq_map):
    key = (product_item, ctq_name)
    if key in ctq_map:
        return ctq_map[key]
    key_global = (None, ctq_name)
    if key_global in ctq_map:
        return ctq_map[key_global]
    return None


@loss_bp.route('/')
@cache.cached(timeout=60, key_prefix='loss_analysis_view', query_string=True)
def loss_analysis():
    item_filter = request.args.get('item', '').strip()
    ctq_filter = request.args.get('ctq', '').strip()
    base_query = LossResult.query
    if item_filter:
        base_query = base_query.filter(LossResult.product_item == item_filter)
    if ctq_filter:
        base_query = base_query.filter(LossResult.ctq_name == ctq_filter)

    total_visible_loss = base_query.with_entities(func.sum(LossResult.batch_total_loss)).scalar() or 0.0
    total_hidden_loss = base_query.with_entities(
        func.sum(LossResult.batch_total_loss_with_hidden - LossResult.batch_total_loss)
    ).scalar() or 0.0
    total_batches = base_query.with_entities(func.count(func.distinct(LossResult.batch_no))).scalar() or 0
    non_compliant_batches = base_query.filter(LossResult.is_gb_compliant == "否").with_entities(
        LossResult.batch_no
    ).distinct().count()
    compliant_batches = total_batches - non_compliant_batches
    compliance_rate = (compliant_batches / total_batches * 100) if total_batches > 0 else 100.0
    avg_unit_loss = base_query.with_entities(func.avg(LossResult.unit_expected_loss)).scalar() or 0.0

    ctq_loss_query = base_query.with_entities(
        LossResult.ctq_name,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).group_by(LossResult.ctq_name).all()
    ctq_loss_data = []
    ctq_names_list = []
    for r in ctq_loss_query:
        name = r[0]
        if name and str(name).strip() not in ('', 'None', 'null'):
            clean_name = str(name).strip()
            ctq_loss_data.append({"name": clean_name, "value": float(r[1] or 0)})
            if clean_name not in ctq_names_list:
                ctq_names_list.append(clean_name)

    item_loss_query = base_query.with_entities(
        LossResult.product_item,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).group_by(LossResult.product_item).all()
    item_labels = []
    item_values = []
    for r in item_loss_query:
        name = r[0]
        if name and str(name).strip() not in ('', 'None', 'null', '[]', 'NoneType'):
            item_labels.append(str(name).strip())
            item_values.append(float(r[1] or 0))

    line_loss_query = base_query.with_entities(
        LossResult.product_line,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).group_by(LossResult.product_line).all()
    line_labels = [str(r[0]).strip() for r in line_loss_query if r[0] and str(r[0]).strip() not in ('', 'None')]
    line_values = [float(r[1] or 0) for r in line_loss_query if r[0] and str(r[0]).strip() not in ('', 'None')]

    process_loss_query = base_query.with_entities(
        CTQConfig.process_link,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).join(LossResult, CTQConfig.ctq_id == LossResult.ctq_id).group_by(CTQConfig.process_link).all()
    process_name_list = [str(r[0]).strip() for r in process_loss_query if
                         r[0] and str(r[0]).strip() not in ('', 'None')]
    process_loss_data = [float(r[1] or 0) for r in process_loss_query if r[0] and str(r[0]).strip() not in ('', 'None')]

    # 周趋势（保持原有周聚合，不修改）
    weekly_trend = base_query.with_entities(
        LossResult.production_week,
        func.sum(LossResult.batch_total_loss).label('total_loss')
    ).filter(LossResult.production_week != None).group_by(LossResult.production_week).order_by(
        LossResult.production_week).all()
    trend_weeks = [f"第{w[0]}周" for w in weekly_trend if w[0] is not None]
    trend_losses = [float(w[1] or 0) for w in weekly_trend if w[0] is not None]

    # ========== 优化后的过程能力分析 ==========
    capability_list = []
    calc = TaguchiQLFCore()
    item_ctq_pairs = base_query.with_entities(
        LossResult.product_item, LossResult.ctq_id, LossResult.ctq_name,
    ).filter(LossResult.product_item != None).group_by(LossResult.product_item, LossResult.ctq_id).all()

    if item_ctq_pairs:
        ctq_ids = list({ctq_id for _, ctq_id, _ in item_ctq_pairs})
        ctq_configs = {c.ctq_id: c for c in CTQConfig.query.filter(CTQConfig.ctq_id.in_(ctq_ids)).all()}
        for item, ctq_id, ctq_name in item_ctq_pairs:
            if not item or str(item).strip() in ('', 'None'):
                continue
            prod_vals = db.session.query(ProductionData.measured_value).filter(
                ProductionData.product_item == item,
                ProductionData.ctq_id == ctq_id
            ).all()
            if len(prod_vals) < 2:
                continue
            y_values = np.array([v[0] for v in prod_vals if v[0] is not None])
            if len(y_values) < 2:
                continue
            ctq_config = ctq_configs.get(ctq_id)
            if not ctq_config:
                continue
            usl = ctq_config.usl
            lsl = ctq_config.lsl
            target_m = ctq_config.target_m
            ppk = calc.calc_ppk(y_values, usl, lsl)
            cpm = calc.calc_cpm(y_values, usl, lsl, target_m)
            capability_list.append({
                "product_item": str(item).strip(),
                "ctq_name": ctq_name,
                "ppk": round(ppk, 4) if not np.isnan(ppk) and ppk is not None else "-",
                "cpm": round(cpm, 4) if not np.isnan(cpm) and cpm is not None else "-",
                "data_count": len(y_values)
            })

    spc_alerts = generate_spc_alerts()

    return render_template('loss_analysis.html',
                           active_page='loss',
                           total_visible_loss=total_visible_loss,
                           total_hidden_loss=total_hidden_loss,
                           compliance_rate=compliance_rate,
                           avg_unit_loss=avg_unit_loss,
                           ctq_loss_data=ctq_loss_data,
                           ctq_names_list=ctq_names_list,
                           item_labels=item_labels,
                           item_values=item_values,
                           line_labels=line_labels,
                           line_values=line_values,
                           process_name_list=process_name_list,
                           process_loss_data=process_loss_data,
                           trend_weeks=trend_weeks,
                           trend_losses=trend_losses,
                           capability_list=capability_list,
                           spc_alerts=spc_alerts)


@loss_bp.route('/api/loss_data')
def api_loss_data():
    draw = request.args.get('draw', type=int)
    start = request.args.get('start', type=int, default=0)
    length = request.args.get('length', type=int, default=25)
    search_value = request.args.get('search[value]', '').strip()
    query = LossResult.query
    if search_value:
        query = query.filter(
            db.or_(
                LossResult.batch_no.ilike(f'%{search_value}%'),
                LossResult.product_item.ilike(f'%{search_value}%'),
                LossResult.ctq_name.ilike(f'%{search_value}%')
            )
        )
    total = query.count()
    order_col_idx = request.args.get('order[0][column]', type=int)
    if order_col_idx is not None:
        col_name = request.args.get(f'columns[{order_col_idx}][data]')
        order_dir = request.args.get('order[0][dir]', 'asc')
        if col_name and hasattr(LossResult, col_name):
            column = getattr(LossResult, col_name)
            if order_dir == 'desc':
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column.asc())
    else:
        query = query.order_by(LossResult.id.desc())
    items = query.offset(start).limit(length).all()
    data = []
    for item in items:
        data.append({
            'batch_no': item.batch_no,
            'product_item': item.product_item or '-',
            'ctq_name': item.ctq_name,
            'measured_value_mean': round(item.measured_value_mean, 4) if item.measured_value_mean is not None else '-',
            'measured_value_std': round(item.measured_value_std, 4) if item.measured_value_std is not None and item.sample_count > 1 else '-',
            'sample_count': item.sample_count,
            'std_source': item.std_source,
            'unit_expected_loss': round(item.unit_expected_loss, 6) if item.unit_expected_loss is not None else '-',
            'batch_total_loss': round(item.batch_total_loss, 2) if item.batch_total_loss is not None else '-',
            'is_gb_compliant': item.is_gb_compliant
        })
    return jsonify({
        'draw': draw,
        'recordsTotal': total,
        'recordsFiltered': total,
        'data': data
    })


@loss_bp.route('/calc', methods=['POST'])
def calc_loss():
    try:
        use_pooled = request.args.get('pooled', 'false').lower() == 'true'
        ctq_map = build_ctq_map()
        calc = TaguchiQLFCore()
        calc_time = datetime.now()

        group_data = defaultdict(list)
        group_meta = {}
        offset = 0
        while True:
            chunk = ProductionData.query.order_by(ProductionData.id).offset(offset).limit(FETCH_BATCH_SIZE).all()
            if not chunk:
                break
            for pd in chunk:
                item = normalize_product_item(pd.product_item)
                key = (item, pd.batch_no, pd.ctq_name)
                group_data[key].append(pd.measured_value)
                if key not in group_meta:
                    group_meta[key] = {
                        'batch_no': pd.batch_no,
                        'product_item': item,
                        'produce_date': pd.produce_date,
                        'product_line': pd.product_line,
                        'production_week': pd.production_week,
                        'production_month': pd.production_month,
                        'batch_quantity': pd.batch_quantity or 0,
                    }
            offset += FETCH_BATCH_SIZE

        if not group_data:
            return jsonify({"code": 400, "msg": "❌ 暂无生产数据"})

        pooled_std = {}
        if use_pooled:
            ctq_pool_group = defaultdict(list)
            for (item, batch, ctq_name), values in group_data.items():
                ctq_pool_group[(item, ctq_name)].append(np.array(values))
            for (item, ctq_name), groups in ctq_pool_group.items():
                pooled = calc.calculate_pooled_std(groups)
                if pooled == 0 and groups:
                    all_vals = np.concatenate(groups)
                    pooled = np.std(all_vals, ddof=1) if len(all_vals) > 1 else 0.0
                pooled_std[(item, ctq_name)] = pooled

        insert_list = []
        skipped_ctqs = set()

        for (item, batch_no, ctq_name), y_list in group_data.items():
            ctq = find_ctq_config(item, ctq_name, ctq_map)
            if not ctq:
                skipped_ctqs.add(f"{item or '通用'}/{ctq_name}")
                continue

            y_array = np.array(y_list)
            mu = np.mean(y_array)
            sigma_batch = np.std(y_array, ddof=1) if len(y_array) > 1 else 0.0
            sample_count = len(y_array)

            if use_pooled and (item, ctq_name) in pooled_std:
                std_used = pooled_std[(item, ctq_name)]
                std_source = "pooled_by_item"
                if std_used == 0:
                    std_used = sigma_batch
                    std_source = "batch_fallback"
            else:
                std_used = sigma_batch
                std_source = "batch"

            asymmetric = (ctq.asymmetric_loss == "是")
            if asymmetric:
                k1 = ctq.k_lower
                k2 = ctq.k_upper
                if k1 == 0 or k2 == 0:
                    delta_upper = ctq.usl - ctq.target_m
                    delta_lower = ctq.target_m - ctq.lsl
                    if delta_upper > 0 and k2 == 0:
                        k2 = calc.calc_k_value_for_side(ctq.feature_type, ctq.a_upper, delta_upper, ctq.fmea_severity)
                    if delta_lower > 0 and k1 == 0:
                        k1 = calc.calc_k_value_for_side(ctq.feature_type, ctq.a_lower, delta_lower, ctq.fmea_severity)
                if k1 == 0 and k2 == 0:
                    skipped_ctqs.add(f"{item or '通用'}/{ctq_name} (K值无效)")
                    continue
            else:
                k = ctq.k_upper
                if k == 0:
                    k = calc.calc_k_value(ctq.feature_type, ctq.a0, ctq.delta0, ctq.a, ctq.delta, ctq.fmea_severity)
                if k == 0:
                    skipped_ctqs.add(f"{item or '通用'}/{ctq_name} (K值无效)")
                    continue
                k1 = k2 = k

            feature = ctq.feature_type
            m = ctq.target_m
            try:
                if feature == "nominal":
                    if asymmetric:
                        if std_used == 0:
                            k_avg = (k1 + k2) / 2
                            unit_loss = k_avg * ((mu - m) ** 2)
                        else:
                            unit_loss = calc.expected_loss_nominal_asymmetric(mu, std_used, m, k1, k2)
                    else:
                        unit_loss = calc.expected_loss_nominal_symmetric(mu, std_used, m, k1)
                elif feature == "smaller":
                    unit_loss = calc.expected_loss_smaller(mu, std_used, k1)
                else:
                    if std_used == 0 and mu > 0:
                        unit_loss = calc._calc_larger_loss(np.array([mu]), k1)[0]
                    else:
                        unit_loss = calc.expected_loss_larger(mu, std_used, k1)
            except Exception:
                if feature == "nominal":
                    unit_loss = k1 * (std_used ** 2 + (mu - m) ** 2)
                elif feature == "smaller":
                    unit_loss = k1 * (mu ** 2 + std_used ** 2)
                else:
                    unit_loss = k1 / (mu ** 2) if mu > 0 else 0.0

            if unit_loss is None or np.isnan(unit_loss):
                unit_loss = 0.0

            is_compliant = "是"
            for y in y_list:
                if (ctq.usl and y > ctq.usl) or (ctq.lsl and y < ctq.lsl):
                    is_compliant = "否"
                    break

            if ctq.process_link in ("仓储物流", "售后环节"):
                paf = "external_failure"
            else:
                paf = "internal_failure" if is_compliant == "否" else "appraisal"

            meta = group_meta.get((item, batch_no, ctq_name), {})
            batch_qty = meta.get('batch_quantity', 0)
            batch_loss = unit_loss * batch_qty
            produce_date = meta.get('produce_date')
            if produce_date:
                year_month = produce_date.strftime('%Y-%m')
            else:
                year_month = None

            loss_record = LossResult(
                batch_no=batch_no,
                produce_date=produce_date if produce_date else datetime.now().date(),
                product_line=meta.get('product_line', ''),
                product_item=item,
                production_week=meta.get('production_week'),
                production_month=meta.get('production_month'),
                production_year_month=year_month,  # 新增字段
                ctq_id=ctq.ctq_id,
                ctq_name=ctq.ctq_name,
                measured_value_mean=round(mu, 6),
                measured_value_std=round(std_used, 6),
                sample_count=sample_count,
                target_deviation=round(mu - m, 6),
                unit_expected_loss=round(unit_loss, 6),
                batch_quantity=batch_qty,
                batch_total_loss=round(batch_loss, 2),
                batch_total_loss_with_hidden=round(batch_loss * ctq.hidden_loss_coef, 2),
                feature_type=feature,
                process_link=ctq.process_link,
                is_ccp=ctq.is_ccp,
                is_gb_compliant=is_compliant,
                paf_category=paf,
                std_source=std_source,
                calc_time=calc_time,
                ctq_version=ctq.version
            )
            insert_list.append(loss_record)

        LossResult.query.delete()
        db.session.bulk_save_objects(insert_list)
        db.session.commit()

        if skipped_ctqs:
            flash(f"⚠️ 以下CTQ因配置问题被跳过：{', '.join(list(skipped_ctqs)[:5])}" +
                  (f" 等{len(skipped_ctqs)}项" if len(skipped_ctqs)>5 else ""), "warning")

        clear_all_caches()
        pooled_param = 'true' if use_pooled else 'false'
        return jsonify({
            "code": 200,
            "msg": f"✅ 成功计算 {len(insert_list)} 条损失（标准差来源：{'长期合并' if use_pooled else '本批样本'}）",
            "pooled": pooled_param
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": f"❌ 计算失败：{str(e)}"})


@loss_bp.route('/export')
def export_excel_data():
    try:
        all_losses = LossResult.query.order_by(LossResult.id).all()
        if not all_losses:
            flash("❌ 暂无分析结果可导出", "danger")
            return redirect(url_for('loss.loss_analysis'))
        data = []
        for item in all_losses:
            data.append({
                "生产批次号": item.batch_no,
                "生产日期": item.produce_date.strftime('%Y-%m-%d') if item.produce_date else '',
                "生产线": item.product_line,
                "品项": item.product_item,
                "生产周": item.production_week,
                "生产月": item.production_month,
                "CTQ名称": item.ctq_name,
                "样本均值": item.measured_value_mean,
                "样本标准差": item.measured_value_std,
                "样本数": item.sample_count,
                "标准差来源": item.std_source,
                "与目标偏差": item.target_deviation,
                "期望单位损失(元)": item.unit_expected_loss,
                "批次产量": item.batch_quantity,
                "批次总损失(元)": item.batch_total_loss,
                "批次总损失(含隐性)(元)": item.batch_total_loss_with_hidden,
                "特性类型": item.feature_type,
                "生产环节": item.process_link,
                "是否CCP": item.is_ccp,
                "国标合规性": item.is_gb_compliant,
                "PAF分类": item.paf_category,
                "计算时间": item.calc_time.strftime('%Y-%m-%d %H:%M:%S') if item.calc_time else ''
            })
        df = pd.DataFrame(data)
        file = export_excel(df, "质量损失分析结果")
        return send_file(file, download_name=f"质量损失分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", as_attachment=True)
    except Exception as e:
        flash(f"❌ 导出失败：{str(e)}", "danger")
        return redirect(url_for('loss.loss_analysis'))


# 修改为 POST 方法 (P0 安全修复)
@loss_bp.route('/clear', methods=['POST'])
def clear_result():
    try:
        LossResult.query.delete()
        db.session.commit()
        clear_all_caches()
        flash("✅ 所有分析结果已清空", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ 清空失败：{str(e)}", "danger")
    return redirect(url_for('loss.loss_analysis'))