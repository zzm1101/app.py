# routes/target_calc.py
# 目标值优化路由

from flask import Blueprint, render_template, request, jsonify
from models.models import CTQConfig, ProductionData, DecayConfig
from services.target_calc import TargetCalcMethods
from config import TARGET_METHODS
from models.database import db
from utils import normalize_product_item
from extensions import clear_all_caches

target_bp = Blueprint('target', __name__, url_prefix='/target')


@target_bp.route('/')
def target_calc():
    try:
        ctq_query = CTQConfig.query.filter_by(status="启用").all()
        ctq_list = []
        for ctq in ctq_query:
            ctq_list.append({
                "ctq_id": ctq.ctq_id,
                "ctq_name": ctq.ctq_name,
                "product_item": ctq.product_item or '通用',
                "feature_type": ctq.feature_type,
                "target_m": ctq.target_m,
                "usl": ctq.usl,
                "lsl": ctq.lsl,
                "fmea_severity": ctq.fmea_severity,
                "is_ccp": ctq.is_ccp,
                "k_upper": ctq.k_upper,
                "k_lower": ctq.k_lower,
                "asymmetric_loss": ctq.asymmetric_loss
            })
        decay_config = DecayConfig.query.first()
        if not decay_config and ctq_query:
            default_ctq = ctq_query[0]
            decay_config = DecayConfig(ctq_id=default_ctq.ctq_id, ctq_name=default_ctq.ctq_name)
            db.session.add(decay_config)
            db.session.commit()
        return render_template('target_calc.html',
                               active_page='target',
                               ctq_list=ctq_list,
                               target_methods=TARGET_METHODS,
                               decay_config=decay_config)
    except Exception as e:
        return f"页面加载失败：{str(e)}", 500


@target_bp.route('/method_status', methods=['POST'])
def get_method_status():
    try:
        data = request.get_json()
        ctq_id = data.get('ctq_id')
        target_cpk = data.get('target_cpk', 1.33)
        target_cpm = data.get('target_cpm', 1.0)
        if not ctq_id:
            return jsonify({"code": 400, "remark": "请先选择CTQ"})
        ctq_row = CTQConfig.query.get_or_404(ctq_id)
        query = ProductionData.query.filter_by(ctq_id=ctq_id)
        if ctq_row.product_item:
            query = query.filter_by(product_item=ctq_row.product_item)
        production_data = query.all()
        decay_config = DecayConfig.query.filter_by(ctq_id=ctq_id).first()
        calc = TargetCalcMethods(ctq_row=ctq_row, production_data=production_data,
                                 decay_config=decay_config, target_cpk=target_cpk, target_cpm=target_cpm)
        method_status = calc.get_method_available_status()
        return jsonify({"code": 200, "method_status": method_status})
    except Exception as e:
        return jsonify({"code": 500, "remark": f"获取失败：{str(e)}"})


@target_bp.route('/calc', methods=['POST'])
def calc_target():
    try:
        data = request.get_json()
        ctq_id = data.get('ctq_id')
        method_key = data.get('method_key')
        target_cpk = float(data.get('target_cpk', 1.33))
        tendency = float(data.get('tendency', 0.5))
        target_cpm = float(data.get('target_cpm', 1.0))
        if not ctq_id or not method_key:
            return jsonify({"code": 400, "optimal_target": None, "remark": "缺少CTQ或计算方法参数"})
        ctq_row = CTQConfig.query.get_or_404(ctq_id)
        query = ProductionData.query.filter_by(ctq_id=ctq_id)
        if ctq_row.product_item:
            query = query.filter_by(product_item=ctq_row.product_item)
        production_data = query.all()
        decay_config = DecayConfig.query.filter_by(ctq_id=ctq_id).first()
        calc = TargetCalcMethods(ctq_row=ctq_row, production_data=production_data,
                                 decay_config=decay_config, target_cpk=target_cpk,
                                 tendency=tendency, target_cpm=target_cpm)
        result = calc.calc_by_method(method_key)
        return jsonify(result)
    except Exception as e:
        return jsonify(
            {"code": 500, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": f"系统异常：{str(e)}"})


@target_bp.route('/recommend', methods=['POST'])
def recommend_method():
    try:
        data = request.get_json()
        ctq_id = data.get('ctq_id')
        target_cpk = float(data.get('target_cpk', 1.33))
        tendency = float(data.get('tendency', 0.5))
        target_cpm = float(data.get('target_cpm', 1.0))
        if not ctq_id:
            return jsonify({"code": 400, "remark": "请先选择CTQ"})
        ctq_row = CTQConfig.query.get_or_404(ctq_id)
        query = ProductionData.query.filter_by(ctq_id=ctq_id)
        if ctq_row.product_item:
            query = query.filter_by(product_item=ctq_row.product_item)
        production_data = query.all()
        decay_config = DecayConfig.query.filter_by(ctq_id=ctq_id).first()
        calc = TargetCalcMethods(ctq_row=ctq_row, production_data=production_data,
                                 decay_config=decay_config, target_cpk=target_cpk,
                                 tendency=tendency, target_cpm=target_cpm)
        recommend_result = calc.recommend_best_method()
        return jsonify(recommend_result)
    except Exception as e:
        return jsonify({"code": 500, "remark": f"推荐失败：{str(e)}"})


@target_bp.route('/save', methods=['POST'])
def save_target():
    try:
        data = request.get_json()
        ctq_id = data.get('ctq_id')
        new_target = data.get('new_target')
        if not ctq_id or new_target is None:
            return jsonify({"code": 400, "msg": "缺少必要参数"})

        ctq = CTQConfig.query.get_or_404(ctq_id)
        # 关键修复：将接收到的字符串转换为浮点数
        try:
            new_target_float = float(new_target)
        except (ValueError, TypeError):
            return jsonify({"code": 400, "msg": "目标值格式无效，请输入数字"})

        # 校验规格限（确保 lsl 和 usl 不为 None）
        if ctq.lsl is None or ctq.usl is None:
            return jsonify({"code": 400, "msg": "CTQ规格限未配置，无法验证目标值"})

        if not (ctq.lsl <= new_target_float <= ctq.usl):
            return jsonify({"code": 400, "msg": f"目标值 {new_target_float} 超出规格限 [{ctq.lsl}, {ctq.usl}]"})

        ctq.target_m = new_target_float
        ctq.version = ctq.version + 1 if ctq.version else 2
        db.session.commit()
        clear_all_caches()
        return jsonify({"code": 200, "msg": "✅ 目标值保存成功，请重新执行损失计算"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": f"保存失败：{str(e)}"})


@target_bp.route('/decay/save', methods=['POST'])
def save_decay_config():
    try:
        data = request.form
        decay_id = data.get('decay_id')
        ctq_id = data.get('ctq_id')
        if not decay_id or not ctq_id:
            return jsonify({"code": 400, "msg": "缺少参数"})
        decay_config = DecayConfig.query.get_or_404(decay_id)
        ctq = CTQConfig.query.get(ctq_id)
        if not ctq:
            return jsonify({"code": 400, "msg": "CTQ不存在"})
        shelf_life = int(data.get('shelf_life_days', 0))
        if shelf_life <= 0:
            return jsonify({"code": 400, "msg": "保质期天数必须大于0"})
        decay_config.ctq_id = ctq_id
        decay_config.ctq_name = ctq.ctq_name
        decay_config.shelf_life_days = shelf_life
        decay_config.std_cold_temp = float(data.get('std_cold_temp', 4))
        decay_config.actual_avg_temp = float(data.get('actual_avg_temp', 6))
        decay_config.temp_fluctuation = float(data.get('temp_fluctuation', 2))
        decay_config.temp_coef = float(data.get('temp_coef', 0.12))
        decay_config.process_std = float(data.get('process_std', 500000))
        db.session.commit()
        return jsonify({"code": 200, "msg": "✅ 衰减参数保存成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": f"保存失败：{str(e)}"})