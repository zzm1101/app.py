# routes/ctq_manage.py
# CTQ 关键质量特性管理模块

from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify
from models.models import CTQConfig, ProductionData, LossResult
from models.database import db
from config import FEATURE_TYPE, FMEA_SEVERITY_K, PRODUCT_ITEMS
from services.excel_service import generate_ctq_template
from sqlalchemy import func
from datetime import datetime
from extensions import cache, clear_all_caches
import pandas as pd
from io import BytesIO
from utils import normalize_product_item, to_float_or_zero

ctq_bp = Blueprint('ctq', __name__, url_prefix='/ctq')

def get_feature_type_alias_map():
    return {
        '望目': 'nominal', '望目特性': 'nominal', 'nominal': 'nominal',
        '望小': 'smaller', '望小特性': 'smaller', 'smaller': 'smaller',
        '望大': 'larger', '望大特性': 'larger', 'larger': 'larger',
    }

def validate_ctq_params(data):
    ctq_name = data.get('ctq_name', '').strip()
    if not ctq_name:
        return False, "CTQ名称不能为空"
    feature_type = data.get('feature_type')
    if feature_type not in FEATURE_TYPE:
        return False, "特性类型无效"
    usl = to_float_or_zero(data.get('usl'))
    lsl = to_float_or_zero(data.get('lsl'))
    target_m = to_float_or_zero(data.get('target_m'))
    if usl <= lsl:
        return False, "上规格限必须大于下规格限"
    if not (lsl <= target_m <= usl):
        return False, "目标值必须在规格限内"
    asymmetric = data.get('asymmetric_loss') == '是'
    if asymmetric:
        a_upper = to_float_or_zero(data.get('a_upper'))
        a_lower = to_float_or_zero(data.get('a_lower'))
        if a_upper <= 0 or a_lower <= 0:
            return False, "非对称损失必须填写大于0的 A_upper 和 A_lower"
        delta_upper = usl - target_m
        delta_lower = target_m - lsl
        if delta_upper <= 0 or delta_lower <= 0:
            return False, "非对称损失要求目标值严格在规格限内部"
    fmea_severity = int(data.get('fmea_severity', 5))
    if fmea_severity not in range(1, 11):
        return False, "FMEA严重度必须是1-10之间的整数"
    hidden_loss_coef = to_float_or_zero(data.get('hidden_loss_coef', 1.0))
    if hidden_loss_coef < 0:
        return False, "隐性损失系数不能为负数"
    return True, None

@ctq_bp.route('/')
def ctq_list():
    ctq_list = CTQConfig.query.order_by(CTQConfig.ctq_id).all()
    enable_count = CTQConfig.query.filter_by(status="启用").count()
    ccp_count = CTQConfig.query.filter_by(is_ccp="是").count()
    high_risk_count = CTQConfig.query.filter(CTQConfig.fmea_severity >= 8).count()
    all_items = set()
    for ctq in ctq_list:
        if ctq.product_item:
            all_items.add(ctq.product_item)
    merged_items = sorted(set(PRODUCT_ITEMS) | all_items)
    return render_template('ctq_manage.html',
                           active_page='ctq',
                           ctq_list=ctq_list,
                           feature_type=FEATURE_TYPE,
                           fmea_config=FMEA_SEVERITY_K,
                           product_items=merged_items,
                           enable_count=enable_count,
                           ccp_count=ccp_count,
                           high_risk_count=high_risk_count)

@ctq_bp.route('/add', methods=['POST'])
def ctq_add():
    try:
        data = request.form
        valid, err_msg = validate_ctq_params(data)
        if not valid:
            flash(f'❌ {err_msg}', 'danger')
            return redirect(url_for('ctq.ctq_list'))
        product_item = normalize_product_item(data.get('product_item', ''))
        existing = CTQConfig.query.filter_by(
            product_item=product_item,
            ctq_name=data['ctq_name'].strip()
        ).first()
        if existing:
            flash(f'❌ CTQ【{data["ctq_name"]}】在品项【{product_item or "通用"}】下已存在', 'danger')
            return redirect(url_for('ctq.ctq_list'))
        new_ctq = CTQConfig(
            product_item=product_item,
            ctq_name=data['ctq_name'].strip(),
            feature_type=data['feature_type'],
            process_link=data.get('process_link', '').strip(),
            is_ccp=data['is_ccp'],
            fmea_severity=int(data['fmea_severity']),
            gb_code=data.get('gb_code', '').strip(),
            target_m=to_float_or_zero(data['target_m']),
            usl=to_float_or_zero(data['usl']),
            lsl=to_float_or_zero(data['lsl']),
            delta0=to_float_or_zero(data.get('delta0')),
            delta=to_float_or_zero(data.get('delta')),
            a0=to_float_or_zero(data.get('a0')),
            a=to_float_or_zero(data.get('a')),
            asymmetric_loss=data.get('asymmetric_loss', '否'),
            k_upper=to_float_or_zero(data.get('k_upper')),
            k_lower=to_float_or_zero(data.get('k_lower')),
            a_upper=to_float_or_zero(data.get('a_upper')),
            a_lower=to_float_or_zero(data.get('a_lower')),
            hidden_loss_coef=to_float_or_zero(data.get('hidden_loss_coef', 1.0)),
            status=data.get('status', '启用'),
            version=1
        )
        db.session.add(new_ctq)
        db.session.commit()
        clear_all_caches()
        flash('✅ CTQ配置添加成功', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ 添加失败：{str(e)}', 'danger')
    return redirect(url_for('ctq.ctq_list'))

@ctq_bp.route('/edit/<int:ctq_id>', methods=['POST'])
def ctq_edit(ctq_id):
    try:
        ctq = CTQConfig.query.get_or_404(ctq_id)
        data = request.form
        valid, err_msg = validate_ctq_params(data)
        if not valid:
            flash(f'❌ {err_msg}', 'danger')
            return redirect(url_for('ctq.ctq_list'))
        product_item = normalize_product_item(data.get('product_item', ''))
        new_ctq_name = data['ctq_name'].strip()
        if (product_item != ctq.product_item or new_ctq_name != ctq.ctq_name):
            conflict = CTQConfig.query.filter_by(
                product_item=product_item,
                ctq_name=new_ctq_name
            ).first()
            if conflict and conflict.ctq_id != ctq_id:
                flash(f'❌ CTQ【{new_ctq_name}】在品项【{product_item or "通用"}】下已存在', 'danger')
                return redirect(url_for('ctq.ctq_list'))
        old_version = ctq.version
        ctq.product_item = product_item
        ctq.ctq_name = new_ctq_name
        ctq.feature_type = data['feature_type']
        ctq.process_link = data.get('process_link', '').strip()
        ctq.is_ccp = data['is_ccp']
        ctq.fmea_severity = int(data['fmea_severity'])
        ctq.gb_code = data.get('gb_code', '').strip()
        ctq.target_m = to_float_or_zero(data['target_m'])
        ctq.usl = to_float_or_zero(data['usl'])
        ctq.lsl = to_float_or_zero(data['lsl'])
        ctq.delta0 = to_float_or_zero(data.get('delta0'))
        ctq.delta = to_float_or_zero(data.get('delta'))
        ctq.a0 = to_float_or_zero(data.get('a0'))
        ctq.a = to_float_or_zero(data.get('a'))
        ctq.asymmetric_loss = data.get('asymmetric_loss', '否')
        ctq.k_upper = to_float_or_zero(data.get('k_upper'))
        ctq.k_lower = to_float_or_zero(data.get('k_lower'))
        ctq.a_upper = to_float_or_zero(data.get('a_upper'))
        ctq.a_lower = to_float_or_zero(data.get('a_lower'))
        ctq.hidden_loss_coef = to_float_or_zero(data.get('hidden_loss_coef', 1.0))
        ctq.status = data.get('status', '启用')
        ctq.version = old_version + 1
        db.session.commit()
        clear_all_caches()
        flash('✅ CTQ配置修改成功', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ 修改失败：{str(e)}', 'danger')
    return redirect(url_for('ctq.ctq_list'))

# 修改为 POST 方法 (P0 安全修复)
@ctq_bp.route('/delete/<int:ctq_id>', methods=['POST'])
def ctq_delete(ctq_id):
    try:
        ctq = CTQConfig.query.get_or_404(ctq_id)
        prod_count = ProductionData.query.filter_by(ctq_id=ctq_id).count()
        loss_count = LossResult.query.filter_by(ctq_id=ctq_id).count()
        if prod_count > 0 or loss_count > 0:
            flash(f'❌ 无法删除：该CTQ已被 {prod_count} 条生产数据和 {loss_count} 条损失结果引用', 'danger')
            return redirect(url_for('ctq.ctq_list'))
        db.session.delete(ctq)
        db.session.commit()
        clear_all_caches()
        flash('✅ 删除成功', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ 删除失败：{str(e)}', 'danger')
    return redirect(url_for('ctq.ctq_list'))

# 修改为 POST 方法 (P0 安全修复)
@ctq_bp.route('/reset', methods=['POST'])
def ctq_reset():
    try:
        if ProductionData.query.count() > 0 or LossResult.query.count() > 0:
            flash('❌ 重置失败：请先清空所有生产数据和损失结果后再执行重置操作', 'danger')
            return redirect(url_for('ctq.ctq_list'))
        CTQConfig.query.delete()
        db.session.commit()
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
        clear_all_caches()
        flash('✅ 已重置为默认国标CTQ配置（通用配置）', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ 重置失败：{str(e)}', 'danger')
    return redirect(url_for('ctq.ctq_list'))

@ctq_bp.route('/template/download')
def download_template():
    try:
        file = generate_ctq_template()
        return send_file(file, download_name="CTQ配置导入模板.xlsx", as_attachment=True)
    except Exception as e:
        flash(f'❌ 模板生成失败：{str(e)}', 'danger')
        return redirect(url_for('ctq.ctq_list'))

@ctq_bp.route('/upload', methods=['POST'])
def upload_ctq():
    file = request.files.get('excel_file')
    if not file:
        flash("❌ 请选择要上传的Excel文件", "danger")
        return redirect(url_for('ctq.ctq_list'))
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        flash("文件过大，请限制在10MB以内", "danger")
        return redirect(url_for('ctq.ctq_list'))
    try:
        df = pd.read_excel(file, sheet_name='数据模板')
        required_cols = ['ctq_name', 'feature_type', 'usl', 'lsl', 'target_m']
        if not all(col in df.columns for col in required_cols):
            flash(f"❌ 模板格式错误，必须包含列：{required_cols}", "danger")
            return redirect(url_for('ctq.ctq_list'))
        existing_map = {}
        for ctq in CTQConfig.query.all():
            key = (ctq.product_item, ctq.ctq_name)
            existing_map[key] = ctq
        alias_map = get_feature_type_alias_map()
        insert_list = []
        update_list = []
        errors = []
        for idx, row in df.iterrows():
            line_num = idx + 2
            try:
                product_item = normalize_product_item(row.get('品项'))
                ctq_name = str(row['ctq_name']).strip()
                if not ctq_name:
                    errors.append(f"第{line_num}行：CTQ名称不能为空")
                    continue
                feature_type_raw = str(row['feature_type']).strip().lower()
                feature_type = alias_map.get(feature_type_raw, feature_type_raw)
                if feature_type not in FEATURE_TYPE:
                    errors.append(f"第{line_num}行：特性类型无效")
                    continue
                usl = float(row['usl'])
                lsl = float(row['lsl'])
                target_m = float(row['target_m'])
                if usl <= lsl:
                    errors.append(f"第{line_num}行：USL 必须大于 LSL")
                    continue
                if not (lsl <= target_m <= usl):
                    errors.append(f"第{line_num}行：目标值必须在规格限内")
                    continue
                asymmetric = str(row.get('asymmetric_loss', '否')).strip()
                if asymmetric == '是':
                    a_upper = float(row.get('a_upper', 0)) if pd.notna(row.get('a_upper')) else 0.0
                    a_lower = float(row.get('a_lower', 0)) if pd.notna(row.get('a_lower')) else 0.0
                    if a_upper <= 0 or a_lower <= 0:
                        errors.append(f"第{line_num}行：非对称损失必须填写正数的 A_upper 和 A_lower")
                        continue
                    delta_upper = usl - target_m
                    delta_lower = target_m - lsl
                    if delta_upper <= 0 or delta_lower <= 0:
                        errors.append(f"第{line_num}行：非对称损失要求目标值严格在规格限内部")
                        continue
                key = (product_item, ctq_name)
                if key in existing_map:
                    ctq = existing_map[key]
                    ctq.feature_type = feature_type
                    ctq.process_link = str(row.get('process_link', '')) if pd.notna(row.get('process_link')) else ''
                    ctq.is_ccp = str(row.get('is_ccp', '否'))
                    ctq.fmea_severity = int(row.get('fmea_severity', 5))
                    ctq.gb_code = str(row.get('gb_code', '')) if pd.notna(row.get('gb_code')) else ''
                    ctq.target_m = target_m
                    ctq.usl = usl
                    ctq.lsl = lsl
                    ctq.delta0 = float(row.get('delta0', 0)) if pd.notna(row.get('delta0')) else 0.0
                    ctq.delta = float(row.get('delta', 0)) if pd.notna(row.get('delta')) else 0.0
                    ctq.a0 = float(row.get('a0', 0)) if pd.notna(row.get('a0')) else 0.0
                    ctq.a = float(row.get('a', 0)) if pd.notna(row.get('a')) else 0.0
                    ctq.asymmetric_loss = asymmetric
                    ctq.k_upper = float(row.get('k_upper', 0)) if pd.notna(row.get('k_upper')) else 0.0
                    ctq.k_lower = float(row.get('k_lower', 0)) if pd.notna(row.get('k_lower')) else 0.0
                    ctq.a_upper = a_upper if asymmetric == '是' else 0.0
                    ctq.a_lower = a_lower if asymmetric == '是' else 0.0
                    ctq.hidden_loss_coef = float(row.get('hidden_loss_coef', 1.0)) if pd.notna(row.get('hidden_loss_coef')) else 1.0
                    ctq.status = str(row.get('status', '启用'))
                    ctq.version += 1
                    update_list.append(ctq)
                else:
                    new_ctq = CTQConfig(
                        product_item=product_item,
                        ctq_name=ctq_name,
                        feature_type=feature_type,
                        process_link=str(row.get('process_link', '')) if pd.notna(row.get('process_link')) else '',
                        is_ccp=str(row.get('is_ccp', '否')),
                        fmea_severity=int(row.get('fmea_severity', 5)),
                        gb_code=str(row.get('gb_code', '')) if pd.notna(row.get('gb_code')) else '',
                        target_m=target_m,
                        usl=usl,
                        lsl=lsl,
                        delta0=float(row.get('delta0', 0)) if pd.notna(row.get('delta0')) else 0.0,
                        delta=float(row.get('delta', 0)) if pd.notna(row.get('delta')) else 0.0,
                        a0=float(row.get('a0', 0)) if pd.notna(row.get('a0')) else 0.0,
                        a=float(row.get('a', 0)) if pd.notna(row.get('a')) else 0.0,
                        asymmetric_loss=asymmetric,
                        k_upper=float(row.get('k_upper', 0)) if pd.notna(row.get('k_upper')) else 0.0,
                        k_lower=float(row.get('k_lower', 0)) if pd.notna(row.get('k_lower')) else 0.0,
                        a_upper=a_upper if asymmetric == '是' else 0.0,
                        a_lower=a_lower if asymmetric == '是' else 0.0,
                        hidden_loss_coef=float(row.get('hidden_loss_coef', 1.0)) if pd.notna(row.get('hidden_loss_coef')) else 1.0,
                        status=str(row.get('status', '启用')),
                        version=1
                    )
                    insert_list.append(new_ctq)
            except Exception as e:
                errors.append(f"第{line_num}行处理错误：{str(e)}")
        if errors:
            error_df = pd.DataFrame(errors, columns=['错误信息'])
            error_output = BytesIO()
            with pd.ExcelWriter(error_output, engine='openpyxl') as writer:
                error_df.to_excel(writer, sheet_name='导入错误', index=False)
            error_output.seek(0)
            flash(f'导入失败，共 {len(errors)} 处错误', 'danger')
            return send_file(error_output, download_name='CTQ导入错误报告.xlsx', as_attachment=True)
        for ctq in update_list:
            db.session.add(ctq)
        db.session.add_all(insert_list)
        db.session.commit()
        clear_all_caches()
        flash(f'✅ 成功导入：新增 {len(insert_list)} 条，更新 {len(update_list)} 条 CTQ 配置', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ 文件处理失败：{str(e)}', 'danger')
    return redirect(url_for('ctq.ctq_list'))

@ctq_bp.route('/api/<int:ctq_id>')
def ctq_api(ctq_id):
    ctq = CTQConfig.query.get_or_404(ctq_id)
    return jsonify({
        'ctq_id': ctq.ctq_id,
        'product_item': ctq.product_item or '',
        'ctq_name': ctq.ctq_name,
        'feature_type': ctq.feature_type,
        'feature_name': FEATURE_TYPE[ctq.feature_type]['name'],
        'process_link': ctq.process_link or '',
        'is_ccp': ctq.is_ccp,
        'fmea_severity': ctq.fmea_severity or 5,
        'gb_code': ctq.gb_code or '',
        'target_m': ctq.target_m or 0,
        'usl': ctq.usl or 0,
        'lsl': ctq.lsl or 0,
        'delta0': ctq.delta0 or 0,
        'delta': ctq.delta or 0,
        'a0': ctq.a0 or 0,
        'a': ctq.a or 0,
        'asymmetric_loss': ctq.asymmetric_loss,
        'k_upper': ctq.k_upper or 0,
        'k_lower': ctq.k_lower or 0,
        'a_upper': ctq.a_upper or 0,
        'a_lower': ctq.a_lower or 0,
        'hidden_loss_coef': ctq.hidden_loss_coef or 1.0,
        'formula': FEATURE_TYPE[ctq.feature_type]['formula'],
        'version': ctq.version
    })