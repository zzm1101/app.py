# routes/spc_page.py
from flask import Blueprint, render_template, request, jsonify
from models.models import ProductionData, CTQConfig, SPCRecord
from services.spc_service import compute_control_chart
from services.taguchi_qlf import TaguchiQLFCore
from models.database import db
from datetime import datetime
import json, logging, numpy as np
from collections import defaultdict, namedtuple
from utils import normalize_product_item
from config import Config
from scipy import stats

spc_bp = Blueprint('spc', __name__, url_prefix='/spc')
logger = logging.getLogger(__name__)

MAX_POINTS_FOR_FRONTEND = 3000   # 前端渲染最大点数


@spc_bp.route('/')
def index():
    ctqs = CTQConfig.query.filter_by(status="启用").all()
    ctq_list = [{"ctq_id": c.ctq_id, "ctq_name": c.ctq_name, "product_item": c.product_item or "通用"} for c in ctqs]
    recent_records = SPCRecord.query.order_by(SPCRecord.id.desc()).limit(10).all()
    all_items = db.session.query(ProductionData.product_item).distinct().filter(ProductionData.product_item.isnot(None)).all()
    all_product_items = sorted([i[0] for i in all_items if i[0]])
    return render_template('spc.html', ctq_list=ctq_list, recent_records=recent_records,
                           active_page='spc', all_product_items=all_product_items)


@spc_bp.route('/data')
def get_spc_data():
    ctq_id = request.args.get('ctq_id', type=int)
    chart_type = request.args.get('chart_type', 'auto')
    usl = request.args.get('usl', type=float)
    lsl = request.args.get('lsl', type=float)
    target = request.args.get('target', type=float)
    product_item_param = request.args.get('product_item', '').strip()
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    use_boxcox = request.args.get('use_boxcox', '0') == '1'

    if not ctq_id or not product_item_param:
        return jsonify({"error": "缺少CTQ或品项"}), 400
    ctq = CTQConfig.query.get(ctq_id)
    if not ctq:
        return jsonify({"error": "CTQ不存在"}), 404

    usl = usl if usl is not None else ctq.usl
    lsl = lsl if lsl is not None else ctq.lsl
    target = target if target is not None else ctq.target_m

    norm_item = normalize_product_item(product_item_param)
    base_query = ProductionData.query.filter(
        ProductionData.ctq_id == ctq_id,
        ProductionData.ctq_name == ctq.ctq_name,
        ProductionData.product_item == norm_item
    )
    if start_date:
        base_query = base_query.filter(ProductionData.produce_date >= start_date)
    if end_date:
        base_query = base_query.filter(ProductionData.produce_date <= end_date)

    total = base_query.count()
    if total == 0:
        return jsonify({"error": f"未找到品项【{product_item_param}】下 CTQ【{ctq.ctq_name}】的生产数据"}), 404

    MAX_BATCHES = Config.MAX_BATCHES_FOR_SPC
    MAX_SAMPLES = Config.MAX_SAMPLES_FOR_SPC

    if total > MAX_SAMPLES and not start_date and not end_date:
        all_data = base_query.order_by(ProductionData.produce_date.desc()).limit(MAX_SAMPLES).all()
        warning = f"数据量超过{MAX_SAMPLES}条，仅分析最新的{len(all_data)}条"
    else:
        if not start_date and not end_date:
            recent = db.session.query(ProductionData.batch_no).filter(
                ProductionData.ctq_id == ctq_id, ProductionData.product_item == norm_item
            ).distinct().order_by(ProductionData.produce_date.desc()).limit(MAX_BATCHES).subquery()
            all_data = base_query.filter(ProductionData.batch_no.in_(recent)).order_by(ProductionData.produce_date, ProductionData.batch_no).all()
            warning = None
        else:
            all_data = base_query.order_by(ProductionData.produce_date, ProductionData.batch_no).all()
            warning = None

    if ctq.feature_type == 'larger':
        original_len = len(all_data)
        all_data = [d for d in all_data if d.measured_value > 0]
        if not all_data:
            return jsonify({"error": "实测值无正数，无法分析望大特性"}), 400
        if len(all_data) < original_len:
            logger.warning(f"过滤掉 {original_len - len(all_data)} 条非正值数据")

    boxcox_lambda = None
    if use_boxcox:
        values = np.array([d.measured_value for d in all_data])
        if np.any(values <= 0):
            return jsonify({"error": "Box-Cox变换要求所有实测值 > 0"}), 400
        try:
            transformed_vals, boxcox_lambda = stats.boxcox(values)
        except Exception as e:
            return jsonify({"error": f"Box-Cox变换失败: {str(e)}"}), 400

        def boxcox_val(x):
            if x is None: return None
            try:
                return stats.boxcox(np.array([x]), lmbda=boxcox_lambda)[0][0]
            except:
                return None

        usl, lsl, target = boxcox_val(usl), boxcox_val(lsl), boxcox_val(target)
        DPoint = namedtuple('DPoint', ['batch_no', 'produce_date', 'measured_value', 'product_item', 'ctq_name'])
        all_data = [DPoint(d.batch_no, d.produce_date, tv, d.product_item, d.ctq_name)
                    for d, tv in zip(all_data, transformed_vals)]

    rules_str = request.args.get('rules', '')
    try:
        active_rules = [int(r.strip()) for r in rules_str.split(',') if r.strip()] if rules_str else list(range(1,9))
    except:
        active_rules = list(range(1,9))

    result = compute_control_chart(all_data, usl, lsl, target, chart_type=chart_type, rules_active=active_rules)
    if "error" in result:
        return jsonify(result), 400

    real_all = [d.measured_value for d in all_data]
    real_mean = np.mean(real_all)
    real_std = np.std(real_all, ddof=1) if len(real_all) > 1 else 0

    # 抽样保护前端
    if len(real_all) > MAX_POINTS_FOR_FRONTEND:
        idx = np.linspace(0, len(real_all)-1, MAX_POINTS_FOR_FRONTEND, dtype=int)
        result['all_values'] = [round(real_all[i], 4) for i in idx]
        result['warning'] = (result.get('warning', '') + ' 数据已抽样显示').strip()
    else:
        result['all_values'] = [round(v, 4) for v in real_all]

    result['mean'] = round(real_mean, 4)
    result['std_overall'] = round(real_std, 6)
    result['ctq_name'] = f"{product_item_param} - {ctq.ctq_name}"
    result['time_range'] = f"{result['dates'][0]} ~ {result['dates'][-1]}" if result.get('dates') else ""
    if warning:
        result['warning'] = (result.get('warning', '') + ' ' + warning).strip()

    try:
        rec = SPCRecord(ctq_id=ctq_id, product_item=product_item_param, chart_type=chart_type,
                        analysis_time=datetime.now(), result_json=json.dumps(result, ensure_ascii=False))
        db.session.add(rec)
        db.session.commit()
    except Exception as e:
        logger.error(f"SPC历史记录保存失败: {e}")

    return jsonify(result)


@spc_bp.route('/capability_trend')
def capability_trend():
    ctq_id = request.args.get('ctq_id', type=int)
    product_item_param = request.args.get('product_item', '').strip()
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    granularity = request.args.get('granularity', 'week')
    use_boxcox = request.args.get('use_boxcox', '0') == '1'

    if not ctq_id or not product_item_param:
        return jsonify({"error": "缺少CTQ或品项"}), 400
    ctq = CTQConfig.query.get(ctq_id)
    if not ctq:
        return jsonify({"error": "CTQ不存在"}), 404

    norm_item = normalize_product_item(product_item_param)
    query = ProductionData.query.filter(
        ProductionData.ctq_id == ctq_id,
        ProductionData.product_item == norm_item
    )
    if start_date:
        query = query.filter(ProductionData.produce_date >= start_date)
    if end_date:
        query = query.filter(ProductionData.produce_date <= end_date)

    rows = query.order_by(ProductionData.produce_date).all()
    if not rows:
        return jsonify({"error": "无数据"}), 404

    usl, lsl, target = ctq.usl, ctq.lsl, ctq.target_m

    if use_boxcox:
        values = np.array([r.measured_value for r in rows])
        if np.any(values <= 0):
            return jsonify({"error": "Box-Cox变换要求所有实测值 > 0"}), 400
        try:
            transformed_vals, lmbda = stats.boxcox(values)
        except Exception as e:
            return jsonify({"error": f"Box-Cox变换失败: {str(e)}"}), 400

        def boxcox_val(x):
            try:
                return stats.boxcox(np.array([x]), lmbda=lmbda)[0][0]
            except:
                return None
        usl, lsl, target = boxcox_val(usl), boxcox_val(lsl), boxcox_val(target)

        TRow = namedtuple('TRow', ['batch_no', 'produce_date', 'measured_value'])
        rows = [TRow(r.batch_no, r.produce_date, tv) for r, tv in zip(rows, transformed_vals)]

    calc = TaguchiQLFCore()
    groups = defaultdict(list)
    group_label = {}
    for row in rows:
        if granularity == 'batch':
            key = row.batch_no
            label = f"{row.batch_no}<br>({row.produce_date})"
        elif granularity == 'week':
            week_num = row.produce_date.isocalendar()[1]
            key = f"{row.produce_date.year}-W{week_num:02d}"
            label = key
        elif granularity == 'month':
            key = row.produce_date.strftime('%Y-%m')
            label = key
        else:
            key = row.batch_no
            label = f"{row.batch_no}<br>({row.produce_date})"
        groups[key].append(row.measured_value)
        group_label[key] = label

    sorted_keys = sorted(groups.keys())
    labels = [group_label[k] for k in sorted_keys]
    ppk_list, cpm_list = [], []
    for key in sorted_keys:
        vals = groups[key]
        if len(vals) < 2:
            ppk_list.append(None)
            cpm_list.append(None)
            continue
        y = np.array(vals)
        ppk = calc.calc_ppk(y, usl, lsl)
        cpm = calc.calc_cpm(y, usl, lsl, target)
        ppk_list.append(round(ppk, 4) if ppk is not None else None)
        cpm_list.append(round(cpm, 4) if cpm is not None else None)

    # 将 NaN 转换为 None (JSON 兼容)
    def clean_nan(lst):
        return [None if isinstance(v, float) and np.isnan(v) else v for v in lst]

    return jsonify({
        "labels": labels,
        "ppk": clean_nan(ppk_list),
        "cpm": clean_nan(cpm_list),
        "target_cpk": 1.33
    })


@spc_bp.route('/history')
def get_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = SPCRecord.query.order_by(SPCRecord.id.desc()).paginate(page=page, per_page=per_page, error_out=False)
    data = []
    for rec in pagination.items:
        ctq = CTQConfig.query.get(rec.ctq_id)
        ctq_name = f"{ctq.product_item or ''} - {ctq.ctq_name}" if ctq else "未知CTQ"
        time_range = ""
        try:
            if rec.result_json:
                res = json.loads(rec.result_json)
                if res.get('dates') and len(res['dates']) > 0:
                    time_range = f"{res['dates'][0]} ~ {res['dates'][-1]}"
        except:
            pass
        data.append({"id": rec.id, "ctq_name": ctq_name, "product_item": rec.product_item,
                     "chart_type": rec.chart_type, "analysis_time": rec.analysis_time.strftime("%Y-%m-%d %H:%M"),
                     "time_range": time_range})
    return jsonify({"data": data, "total": pagination.total, "pages": pagination.pages, "current_page": page})


@spc_bp.route('/history/<int:record_id>')
def get_history_detail(record_id):
    rec = SPCRecord.query.get_or_404(record_id)
    try:
        return jsonify(json.loads(rec.result_json))
    except:
        return jsonify({"error": "记录数据损坏"}), 500