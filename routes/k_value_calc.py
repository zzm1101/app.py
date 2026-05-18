# routes/k_value_calc.py
# K值损失系数计算模块

from flask import Blueprint, render_template, request, jsonify
from models.models import CTQConfig
from services.taguchi_qlf import TaguchiQLFCore
from config import FMEA_SEVERITY_K, FEATURE_TYPE
from models.database import db
from extensions import cache, clear_all_caches
from utils import to_float_or_zero

k_value_bp = Blueprint('k_value', __name__, url_prefix='/k_value')


@k_value_bp.route('/', methods=['GET'])
def k_value_calc():
    ctq_query = CTQConfig.query.filter_by(status="启用").all()
    ctq_list = []
    for ctq in ctq_query:
        ctq_list.append({
            "ctq_id": ctq.ctq_id,
            "ctq_name": ctq.ctq_name,
            "product_item": ctq.product_item or '通用',
            "feature_type": ctq.feature_type,
            "feature_name": FEATURE_TYPE[ctq.feature_type]["name"],
            "target_m": ctq.target_m,
            "usl": ctq.usl,
            "lsl": ctq.lsl,
            "fmea_severity": ctq.fmea_severity,
            "a0": ctq.a0,
            "delta0": ctq.delta0,
            "a": ctq.a,
            "delta": ctq.delta,
            "asymmetric_loss": ctq.asymmetric_loss,
            "k_upper": ctq.k_upper,
            "k_lower": ctq.k_lower,
            "a_upper": ctq.a_upper,
            "a_lower": ctq.a_lower,
            "formula": FEATURE_TYPE[ctq.feature_type]["formula"]
        })
    return render_template('k_value_calc.html',
                           active_page='k_value',
                           ctq_list=ctq_list,
                           fmea_config=FMEA_SEVERITY_K,
                           feature_type=FEATURE_TYPE)


@k_value_bp.route('/calc', methods=['POST'])
def calc_k_value():
    try:
        data = request.get_json()
        ctq_id = data.get('ctq_id')
        if not ctq_id:
            return jsonify({"code": 400, "msg": "缺少CTQ编号"})
        ctq = CTQConfig.query.get_or_404(ctq_id)
        calc = TaguchiQLFCore()
        asymmetric = (ctq.asymmetric_loss == "是")
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        if asymmetric:
            a_upper = safe_float(data.get('a_upper'), ctq.a_upper)
            a_lower = safe_float(data.get('a_lower'), ctq.a_lower)
            m = ctq.target_m
            usl = ctq.usl
            lsl = ctq.lsl
            delta_upper = usl - m if usl and m else 0.0
            delta_lower = m - lsl if m and lsl else 0.0
            if delta_upper <= 0 or delta_lower <= 0:
                return jsonify({"code": 400, "msg": f"非对称CTQ的规格限或目标值设置有误：ΔU={delta_upper}, ΔL={delta_lower}"})
            k_upper = calc.calc_k_value_for_side(
                feature_type=ctq.feature_type,
                a=a_upper,
                delta=delta_upper,
                fmea_severity=ctq.fmea_severity
            )
            k_lower = calc.calc_k_value_for_side(
                feature_type=ctq.feature_type,
                a=a_lower,
                delta=delta_lower,
                fmea_severity=ctq.fmea_severity
            )
            return jsonify({
                "code": 200,
                "k_upper": round(float(k_upper), 6),
                "k_lower": round(float(k_lower), 6),
                "asymmetric": True,
                "delta_upper": round(delta_upper, 4),
                "delta_lower": round(delta_lower, 4)
            })
        else:
            k = calc.calc_k_value(
                feature_type=ctq.feature_type,
                a0=ctq.a0,
                delta0=ctq.delta0,
                a=ctq.a,
                delta=ctq.delta,
                fmea_severity=ctq.fmea_severity
            )
            if k == 0:
                return jsonify({"code": 400, "msg": "K值计算为0，请检查A/Δ或A0/Δ0参数是否为零"})
            return jsonify({
                "code": 200,
                "k": round(float(k), 6),
                "asymmetric": False
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "msg": f"计算失败：{str(e)}"})


@k_value_bp.route('/save', methods=['POST'])
def save_k_value():
    try:
        data = request.get_json()
        ctq_id = data.get('ctq_id')
        if not ctq_id:
            return jsonify({"code": 400, "msg": "缺少CTQ编号"})
        ctq = CTQConfig.query.get_or_404(ctq_id)
        k_upper = to_float_or_zero(data.get('k_upper'))
        k_lower = to_float_or_zero(data.get('k_lower', k_upper))
        if k_upper == 0 and k_lower == 0:
            return jsonify({"code": 400, "msg": "K值不能为0，请先计算"})
        if 'a_upper' in data:
            ctq.a_upper = to_float_or_zero(data['a_upper'])
            ctq.a_lower = to_float_or_zero(data.get('a_lower', data['a_upper']))
        ctq.k_upper = k_upper
        ctq.k_lower = k_lower
        ctq.version = ctq.version + 1 if ctq.version else 2
        db.session.commit()
        clear_all_caches()
        return jsonify({"code": 200, "msg": "✅ K值保存成功，版本号已更新"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": f"保存失败：{str(e)}"})