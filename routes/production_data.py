# routes/production_data.py
# 生产数据管理模块
# 包含：列表展示（服务端分页）、模板下载、模拟数据生成、批量导入、手动增删改查、导出、清空

from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify
from models.models import ProductionData, CTQConfig, LossResult
from models.database import db
from services.excel_service import generate_production_template, generate_mock_production_data, export_excel
from datetime import date, datetime
import pandas as pd
import numpy as np
from io import BytesIO
from extensions import cache, clear_all_caches
from utils import normalize_product_item, to_float_or_zero, is_valid_date

production_bp = Blueprint('production', __name__, url_prefix='/production')

# ---------- 辅助函数 ----------
def validate_measured_value(value, ctq, line_num=None):
    if value is None:
        return False, "实测值不能为空"
    if ctq.usl and ctq.lsl:
        if value < ctq.lsl * 0.5 or value > ctq.usl * 1.5:
            warning = f"实测值 {value} 超出合理范围（规格限 {ctq.lsl}~{ctq.usl}）"
            return True, warning
    return True, None

# ---------- 页面路由 ----------
@production_bp.route('/')
@cache.cached(timeout=30, key_prefix='production_list_view')
def data_list():
    total_count = ProductionData.query.count()
    batch_count = ProductionData.query.with_entities(ProductionData.batch_no).distinct().count()
    product_line_count = ProductionData.query.with_entities(ProductionData.product_line).distinct().count()
    ctq_count = ProductionData.query.with_entities(ProductionData.ctq_id).distinct().count()
    ctq_configs = CTQConfig.query.filter_by(status="启用").order_by(CTQConfig.product_item, CTQConfig.ctq_name).all()
    ctq_dicts = []
    for ctq in ctq_configs:
        ctq_dicts.append({
            'ctq_id': ctq.ctq_id,
            'product_item': ctq.product_item or '',
            'ctq_name': ctq.ctq_name,
        })
    return render_template('production_data.html',
                           active_page='production',
                           total_count=total_count,
                           batch_count=batch_count,
                           product_line_count=product_line_count,
                           ctq_count=ctq_count,
                           ctq_list=ctq_dicts,
                           today=date.today())

# ---------- API：服务端分页数据 ----------
@production_bp.route('/api/data')
def api_production_data():
    draw = request.args.get('draw', type=int)
    start = request.args.get('start', type=int, default=0)
    length = request.args.get('length', type=int, default=25)
    search_value = request.args.get('search[value]', '').strip()
    query = ProductionData.query
    if search_value:
        query = query.filter(
            db.or_(
                ProductionData.batch_no.ilike(f'%{search_value}%'),
                ProductionData.product_item.ilike(f'%{search_value}%'),
                ProductionData.ctq_name.ilike(f'%{search_value}%'),
                ProductionData.inspector.ilike(f'%{search_value}%')
            )
        )
    total = query.count()
    order_col_idx = request.args.get('order[0][column]', type=int)
    if order_col_idx is not None:
        col_name = request.args.get(f'columns[{order_col_idx}][data]')
        order_dir = request.args.get('order[0][dir]', 'asc')
        if col_name and hasattr(ProductionData, col_name):
            column = getattr(ProductionData, col_name)
            if order_dir == 'desc':
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column.asc())
    else:
        query = query.order_by(ProductionData.id.desc())
    items = query.offset(start).limit(length).all()
    data = []
    for item in items:
        data.append({
            'id': item.id,
            'batch_no': item.batch_no,
            'produce_date': item.produce_date.isoformat() if item.produce_date else '',
            'product_line': item.product_line or '',
            'work_shift': item.work_shift or '',
            'product_item': item.product_item or '',
            'ctq_id': item.ctq_id,
            'ctq_name': item.ctq_name,
            'measured_value': item.measured_value,
            'batch_quantity': item.batch_quantity,
            'sample_no': item.sample_no or '',
            'storage_days': item.storage_days or 0,
            'storage_temp': item.storage_temp or 4,
            'inspector': item.inspector or '',
            'create_time': item.create_time.isoformat() if item.create_time else '',
        })
    return jsonify({
        'draw': draw,
        'recordsTotal': total,
        'recordsFiltered': total,
        'data': data
    })

# ---------- 模板下载 ----------
@production_bp.route('/template/download')
def download_template():
    try:
        file = generate_production_template()
        return send_file(file, download_name="酸奶生产数据导入模板.xlsx", as_attachment=True)
    except Exception as e:
        flash(f'❌ 模板生成失败：{str(e)}', 'danger')
        return redirect(url_for('production.data_list'))

# ---------- 模拟数据生成 ----------
@production_bp.route('/mock/generate')
def generate_mock():
    try:
        enabled_ctq_count = CTQConfig.query.filter_by(status="启用").count()
        if enabled_ctq_count == 0:
            flash('❌ 没有启用的CTQ配置', 'danger')
            return redirect(url_for('production.data_list'))
        mock_df = generate_mock_production_data()
        if mock_df.empty:
            flash('⚠️ 未生成任何数据', 'warning')
            return redirect(url_for('production.data_list'))
        new_records = []
        for _, row in mock_df.iterrows():
            produce_date = row['生产日期']
            if isinstance(produce_date, str):
                produce_date = datetime.strptime(produce_date, '%Y-%m-%d').date()
            week = produce_date.isocalendar()[1]
            month = produce_date.month
            year_month = produce_date.strftime('%Y-%m')  # 新增
            new_data = ProductionData(
                batch_no=row['生产批次号'],
                produce_date=produce_date,
                product_line=row.get('生产线', ''),
                work_shift=row.get('生产班次', ''),
                product_item=normalize_product_item(row.get('品项', '')),
                sample_no=row.get('样品编号', ''),
                ctq_id=row['CTQ编号'],
                ctq_name=row['CTQ名称'],
                measured_value=row['实测值'],
                batch_quantity=row.get('批次生产数量', 0),
                storage_days=row.get('存储天数', 0),
                storage_temp=row.get('存储温度(℃)', 4),
                inspector=row.get('检验员', ''),
                production_week=week,
                production_month=month,
                production_year_month=year_month
            )
            new_records.append(new_data)
        db.session.bulk_save_objects(new_records)
        db.session.commit()
        clear_all_caches()
        flash(f'✅ 成功生成{len(new_records)}条模拟生产数据', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ 模拟数据生成失败：{str(e)}', 'danger')
    return redirect(url_for('production.data_list'))

# ---------- 批量导入 ----------
@production_bp.route('/upload', methods=['POST'])
def upload_data():
    file = request.files.get('excel_file')
    if not file:
        flash("❌ 请选择要上传的Excel文件", "danger")
        return redirect(url_for('production.data_list'))
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        flash("文件过大，请限制在10MB以内", "danger")
        return redirect(url_for('production.data_list'))
    try:
        df = pd.read_excel(file)
        required_cols = ["生产批次号", "生产日期", "CTQ名称", "实测值", "批次生产数量"]
        if not all(col in df.columns for col in required_cols):
            flash(f"❌ 数据缺少必要列，必须包含：{required_cols}", "danger")
            return redirect(url_for('production.data_list'))
        ctq_map = {}
        for ctq in CTQConfig.query.filter_by(status="启用").all():
            key = (ctq.product_item, ctq.ctq_name)
            ctq_map[key] = ctq
            if ctq.product_item is None:
                ctq_map[(None, ctq.ctq_name)] = ctq
        new_records = []
        errors = []
        warnings = []
        for idx, row in df.iterrows():
            line_num = idx + 2
            try:
                batch_no = str(row['生产批次号']).strip()
                if not batch_no:
                    errors.append(f"第{line_num}行：生产批次号不能为空")
                    continue
                produce_date = pd.to_datetime(row['生产日期']).date()
                if not is_valid_date(produce_date, allow_future=False):
                    errors.append(f"第{line_num}行：生产日期不能晚于今天")
                    continue
                ctq_name = str(row['CTQ名称']).strip()
                if not ctq_name:
                    errors.append(f"第{line_num}行：CTQ名称不能为空")
                    continue
                product_item_raw = row.get('品项')
                product_item = normalize_product_item(product_item_raw)
                ctq = ctq_map.get((product_item, ctq_name))
                if not ctq:
                    ctq = ctq_map.get((None, ctq_name))
                if not ctq:
                    errors.append(f"第{line_num}行：未找到CTQ配置 [品项:{product_item}] CTQ:{ctq_name}")
                    continue
                measured_value = float(row['实测值'])
                valid, warning = validate_measured_value(measured_value, ctq, line_num)
                if not valid:
                    errors.append(f"第{line_num}行：{warning}")
                    continue
                if warning:
                    warnings.append(f"第{line_num}行：{warning}")
                batch_quantity = int(row.get('批次生产数量', 0))
                if batch_quantity <= 0:
                    errors.append(f"第{line_num}行：批次生产数量必须大于0")
                    continue
                week = produce_date.isocalendar()[1]
                month = produce_date.month
                year_month = produce_date.strftime('%Y-%m')
                new_data = ProductionData(
                    batch_no=batch_no,
                    produce_date=produce_date,
                    product_line=str(row.get('生产线', '')) if pd.notna(row.get('生产线')) else '',
                    work_shift=str(row.get('生产班次', '')) if pd.notna(row.get('生产班次')) else '',
                    product_item=product_item,
                    sample_no=str(row.get('样品编号', '')) if pd.notna(row.get('样品编号')) else '',
                    ctq_id=ctq.ctq_id,
                    ctq_name=ctq.ctq_name,
                    measured_value=measured_value,
                    batch_quantity=batch_quantity,
                    storage_days=int(row.get('存储天数', 0)) if pd.notna(row.get('存储天数')) else 0,
                    storage_temp=float(row.get('存储温度(℃)', 4)) if pd.notna(row.get('存储温度(℃)')) else 4,
                    inspector=str(row.get('检验员', '')) if pd.notna(row.get('检验员')) else '',
                    production_week=week,
                    production_month=month,
                    production_year_month=year_month
                )
                new_records.append(new_data)
            except Exception as e:
                errors.append(f"第{line_num}行处理错误：{str(e)}")
        for w in warnings[:5]:
            flash(w, 'warning')
        if len(warnings) > 5:
            flash(f'... 还有 {len(warnings)-5} 条警告', 'warning')
        if errors:
            for err in errors[:10]:
                flash(err, 'danger')
            if len(errors) > 10:
                flash(f'... 还有 {len(errors)-10} 条错误', 'danger')
            db.session.rollback()
            return redirect(url_for('production.data_list'))
        db.session.bulk_save_objects(new_records)
        db.session.commit()
        clear_all_caches()
        flash(f"✅ 成功导入{len(new_records)}条生产数据", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ 文件导入失败：{str(e)}", "danger")
    return redirect(url_for('production.data_list'))

# ---------- 手动新增 ----------
@production_bp.route('/add', methods=['POST'])
def add_data():
    try:
        data = request.form
        ctq = CTQConfig.query.get_or_404(data['ctq_id'])
        product_item = normalize_product_item(data.get('product_item', ''))
        if ctq.product_item and ctq.product_item != product_item:
            flash(f"❌ 当前CTQ【{ctq.ctq_name}】已绑定品项【{ctq.product_item}】，不能用于品项【{product_item}】", "danger")
            return redirect(url_for('production.data_list'))
        produce_date = datetime.strptime(data['produce_date'], '%Y-%m-%d').date()
        if not is_valid_date(produce_date):
            flash("生产日期不能晚于今天", "danger")
            return redirect(url_for('production.data_list'))
        measured_value = float(data['measured_value'])
        valid, warning = validate_measured_value(measured_value, ctq)
        if not valid:
            flash(f"实测值无效：{warning}", "danger")
            return redirect(url_for('production.data_list'))
        if warning:
            flash(warning, "warning")
        batch_quantity = int(data.get('batch_quantity', 0))
        if batch_quantity <= 0:
            flash("批次生产数量必须大于0", "danger")
            return redirect(url_for('production.data_list'))
        week = produce_date.isocalendar()[1]
        month = produce_date.month
        year_month = produce_date.strftime('%Y-%m')
        new_data = ProductionData(
            batch_no=data['batch_no'].strip(),
            produce_date=produce_date,
            product_line=data.get('product_line', '').strip(),
            work_shift=data.get('work_shift', ''),
            product_item=product_item,
            sample_no=data.get('sample_no', '').strip(),
            ctq_id=ctq.ctq_id,
            ctq_name=ctq.ctq_name,
            measured_value=measured_value,
            batch_quantity=batch_quantity,
            storage_days=int(data.get('storage_days', 0)),
            storage_temp=float(data.get('storage_temp', 4)),
            inspector=data.get('inspector', '').strip(),
            production_week=week,
            production_month=month,
            production_year_month=year_month
        )
        db.session.add(new_data)
        db.session.commit()
        clear_all_caches()
        flash("✅ 生产数据添加成功", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ 添加失败：{str(e)}", "danger")
    return redirect(url_for('production.data_list'))

# ---------- 编辑 ----------
@production_bp.route('/edit/<int:data_id>', methods=['POST'])
def edit_data(data_id):
    try:
        item = ProductionData.query.get_or_404(data_id)
        form = request.form
        ctq = CTQConfig.query.get_or_404(form['ctq_id'])
        product_item = normalize_product_item(form.get('product_item', ''))
        if ctq.product_item and ctq.product_item != product_item:
            flash(f"❌ 当前CTQ已绑定品项【{ctq.product_item}】，不能改为品项【{product_item}】", "danger")
            return redirect(url_for('production.data_list'))
        produce_date = datetime.strptime(form['produce_date'], '%Y-%m-%d').date()
        if not is_valid_date(produce_date):
            flash("生产日期不能晚于今天", "danger")
            return redirect(url_for('production.data_list'))
        measured_value = float(form['measured_value'])
        valid, warning = validate_measured_value(measured_value, ctq)
        if not valid:
            flash(f"实测值无效：{warning}", "danger")
            return redirect(url_for('production.data_list'))
        batch_quantity = int(form.get('batch_quantity', 0))
        if batch_quantity <= 0:
            flash("批次生产数量必须大于0", "danger")
            return redirect(url_for('production.data_list'))
        item.batch_no = form['batch_no'].strip()
        item.produce_date = produce_date
        item.product_line = form.get('product_line', '').strip()
        item.work_shift = form.get('work_shift', '')
        item.product_item = product_item
        item.sample_no = form.get('sample_no', '').strip()
        item.ctq_id = ctq.ctq_id
        item.ctq_name = ctq.ctq_name
        item.measured_value = measured_value
        item.batch_quantity = batch_quantity
        item.storage_days = int(form.get('storage_days', 0))
        item.storage_temp = float(form.get('storage_temp', 4))
        item.inspector = form.get('inspector', '').strip()
        # 重新计算周、月、年月
        item.production_week = produce_date.isocalendar()[1]
        item.production_month = produce_date.month
        item.production_year_month = produce_date.strftime('%Y-%m')
        db.session.commit()
        LossResult.query.filter_by(batch_no=item.batch_no, ctq_id=item.ctq_id).delete()
        db.session.commit()
        clear_all_caches()
        flash("✅ 修改成功，请重新执行损失计算以更新分析结果", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ 修改失败：{str(e)}", "danger")
    return redirect(url_for('production.data_list'))

# ---------- 删除 ----------
@production_bp.route('/delete/<int:data_id>', methods=['POST'])
def delete_data(data_id):
    try:
        data = ProductionData.query.get_or_404(data_id)
        batch_no = data.batch_no
        ctq_id = data.ctq_id
        db.session.delete(data)
        remaining = ProductionData.query.filter_by(batch_no=batch_no, ctq_id=ctq_id).count()
        if remaining == 0:
            LossResult.query.filter_by(batch_no=batch_no, ctq_id=ctq_id).delete()
        db.session.commit()
        clear_all_caches()
        # 判断是否是 AJAX 请求（用于异步删除）
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True})
        flash("✅ 生产数据删除成功", "success")
        return redirect(url_for('production.data_list'))
    except Exception as e:
        db.session.rollback()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "error": str(e)}), 500
        flash(f"❌ 生产数据删除失败：{str(e)}", "danger")
        return redirect(url_for('production.data_list'))

# ---------- 清空全部 ----------
@production_bp.route('/clear', methods=['POST'])
def clear_data():
    try:
        count = ProductionData.query.count()
        if count > 0:
            flash(f"⚠️ 将删除 {count} 条生产数据及所有损失结果", "warning")
        ProductionData.query.delete()
        LossResult.query.delete()
        db.session.commit()
        # ========== 新增下面这一行 ==========
        cache.delete('production_list_view')  # 删除生产列表页面的缓存
        clear_all_caches()
        flash("✅ 所有生产数据及损失结果已清空", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ 清空失败：{str(e)}", "danger")
    return redirect(url_for('production.data_list'))

# ---------- 导出 ----------
@production_bp.route('/export')
def export_data():
    try:
        data_list = ProductionData.query.all()
        if not data_list:
            flash("❌ 暂无生产数据可导出", "danger")
            return redirect(url_for('production.data_list'))
        export_data = []
        for item in data_list:
            export_data.append({
                "生产批次号": item.batch_no,
                "生产日期": item.produce_date,
                "生产线": item.product_line,
                "生产班次": item.work_shift,
                "品项": item.product_item,
                "样品编号": item.sample_no,
                "CTQ编号": item.ctq_id,
                "CTQ名称": item.ctq_name,
                "实测值": item.measured_value,
                "批次生产数量": item.batch_quantity,
                "存储天数": item.storage_days,
                "存储温度(℃)": item.storage_temp,
                "检验员": item.inspector,
                "录入时间": item.create_time.strftime('%Y-%m-%d %H:%M:%S')
            })
        df = pd.DataFrame(export_data)
        file = export_excel(df, "生产数据明细")
        return send_file(file, download_name=f"酸奶工厂生产数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", as_attachment=True)
    except Exception as e:
        flash(f"❌ 导出失败：{str(e)}", "danger")
        return redirect(url_for('production.data_list'))