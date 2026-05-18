# routes/improvement.py
# 改进优先级与智能建议模块

from flask import Blueprint, render_template, send_file
from models.models import LossResult, CTQConfig, ProductionData, db
from services.taguchi_qlf import TaguchiQLFCore
from services.excel_service import export_excel
from sqlalchemy import func
import pandas as pd
import numpy as np
from datetime import datetime
from extensions import cache, clear_all_caches

improvement_bp = Blueprint('improvement', __name__, url_prefix='/improvement')


def get_cpk_from_raw_data(product_item, ctq_id, usl, lsl):
    """从原始生产数据计算真实CPK（而非批次均值）"""
    query = ProductionData.query.filter_by(ctq_id=ctq_id)
    if product_item:
        query = query.filter_by(product_item=product_item)
    values = [d.measured_value for d in query.all() if d.measured_value is not None]
    if len(values) < 2:
        return None
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return None
    cpu = (usl - mean) / (3 * std) if usl else None
    cpl = (mean - lsl) / (3 * std) if lsl else None
    if cpu is None or cpl is None:
        return None
    return min(cpu, cpl)


def get_priority_list(total_loss):
    priority_list = []
    ctq_configs = CTQConfig.query.filter_by(status="启用").all()
    config_map = {}
    for ctq in ctq_configs:
        item = ctq.product_item if ctq.product_item else '__global__'
        config_map[(item, ctq.ctq_name)] = ctq

    groups = db.session.query(
        LossResult.product_item,
        LossResult.ctq_name,
        LossResult.ctq_id,
        func.sum(LossResult.batch_total_loss_with_hidden).label('total_loss')
    ).group_by(LossResult.product_item, LossResult.ctq_name).all()

    for item, ctq_name, ctq_id, loss_sum in groups:
        ctq_config = config_map.get((item, ctq_name)) or config_map.get(('__global__', ctq_name))
        if not ctq_config:
            continue
        severity = ctq_config.fmea_severity or 5
        usl = ctq_config.usl
        lsl = ctq_config.lsl
        is_ccp = (ctq_config.is_ccp == "是")
        cpk = get_cpk_from_raw_data(item, ctq_id, usl, lsl)
        if cpk is None:
            cpk_rank = 3
            cpk_display = "数据不足"
        else:
            cpk_display = round(cpk, 4)
            if cpk < 1.0:
                cpk_rank = 3
            elif cpk < 1.33:
                cpk_rank = 2
            else:
                cpk_rank = 1
        if total_loss == 0:
            loss_rank = 1
        else:
            ratio = loss_sum / total_loss
            if ratio >= 0.4:
                loss_rank = 3
            elif ratio >= 0.15:
                loss_rank = 2
            else:
                loss_rank = 1
        if severity >= 8:
            severity_rank = 3
        elif severity >= 5:
            severity_rank = 2
        else:
            severity_rank = 1
        priority_score = loss_rank + severity_rank + cpk_rank
        if is_ccp:
            priority_score += 1
        if priority_score >= 8:
            level = "极高优先级"
            level_class = "danger"
        elif priority_score >= 6:
            level = "高优先级"
            level_class = "warning"
        elif priority_score >= 4:
            level = "中优先级"
            level_class = "primary"
        else:
            level = "低优先级"
            level_class = "success"
        priority_list.append({
            "product_item": item if item else "通用",
            "ctq_name": ctq_name,
            "total_loss": round(loss_sum, 2),
            "fmea_severity": severity,
            "cpk": cpk_display,
            "priority_score": priority_score,
            "priority_level": level,
            "level_class": level_class,
            "process_link": ctq_config.process_link or "",
            "is_ccp": "是" if is_ccp else "否",
        })
    priority_list.sort(key=lambda x: x["priority_score"], reverse=True)
    return priority_list


def generate_suggestions(priority_list, total_loss, compliance_rate):
    suggestions = []
    if not priority_list:
        suggestions.append({"title": "暂无数据", "content": "请先执行损失计算后再查看改进建议。"})
        return suggestions
    top = priority_list[0]
    suggestions.append({
        "title": "🔴 首要改进目标",
        "content": (f"【{top['product_item']} - {top['ctq_name']}】总损失 {top['total_loss']} 元，"
                   f"严重度 {top['fmea_severity']} 级，CPK={top['cpk']}。"
                   f"建议立即成立专项改善小组，运用FMEA重新评估控制点，优化目标值（可使用目标值优化模块）。")
    })
    low_cpk_ctqs = [item for item in priority_list
                    if isinstance(item['cpk'], (int, float)) and item['cpk'] < 1.33]
    if low_cpk_ctqs:
        names = "、".join([f"{c['product_item']}-{c['ctq_name']}" for c in low_cpk_ctqs[:3]])
        suggestions.append({
            "title": "⚠️ 过程能力不足",
            "content": (f"{names} 的CPK低于1.33，表明过程波动较大。"
                       f"建议进行MSA和过程FMEA，排查设备、原料、操作方法的影响，并实施SPC实时监控。")
        })
    if total_loss > 100000:
        suggestions.append({
            "title": "💰 高额质量损失",
            "content": (f"累计总质量损失已超过10万元（{total_loss:,.2f}元）。"
                       f"建议从隐性损失源头治理，重点改善冷链运输温度和灌装净含量等CCP控制点。")
        })
    if compliance_rate < 99.5:
        suggestions.append({
            "title": "📉 合规率偏低",
            "content": (f"当前国标合规率仅为{compliance_rate:.2f}%，低于99.5%的良好目标。"
                       f"请导出损失明细，筛选不合规批次进行根本原因分析。")
        })
    ccp_ctqs = [item for item in priority_list
                if item['is_ccp'] == '是' and isinstance(item['cpk'], (int, float)) and item['cpk'] < 1.33]
    if ccp_ctqs:
        ccp_names = "、".join([f"{c['product_item']}-{c['ctq_name']}" for c in ccp_ctqs[:3]])
        suggestions.append({
            "title": "🛑 CCP关键控制点能力不足",
            "content": (f"{ccp_names} 属于HACCP计划中的关键控制点，但过程能力不足。"
                       f"请立即复核监控记录和控制限，必要时调整工艺参数或增加检测频次。")
        })
    return suggestions


@improvement_bp.route('/')
@cache.cached(timeout=60, key_prefix='improvement_v2')
def improvement_list():
    total_loss = db.session.query(func.sum(LossResult.batch_total_loss_with_hidden)).scalar() or 0.0
    total_batches = db.session.query(func.count(func.distinct(LossResult.batch_no))).scalar() or 0
    non_compliant_batches = db.session.query(LossResult.batch_no).filter(
        LossResult.is_gb_compliant == "否"
    ).distinct().count()
    compliant_batches = total_batches - non_compliant_batches
    compliance_rate = (compliant_batches / total_batches * 100) if total_batches > 0 else 100.0
    priority_list = get_priority_list(total_loss)
    suggestions = generate_suggestions(priority_list, total_loss, compliance_rate)
    return render_template('improvement.html',
                           active_page='improvement',
                           priority_list=priority_list,
                           total_loss=total_loss,
                           compliance_rate=compliance_rate,
                           suggestions=suggestions)


@improvement_bp.route('/report/export')
def export_report():
    try:
        total_loss = db.session.query(func.sum(LossResult.batch_total_loss_with_hidden)).scalar() or 0.0
        total_batches = db.session.query(func.count(func.distinct(LossResult.batch_no))).scalar() or 0
        non_compliant_batches = db.session.query(LossResult.batch_no).filter(
            LossResult.is_gb_compliant == "否"
        ).distinct().count()
        compliant_batches = total_batches - non_compliant_batches
        compliance_rate = (compliant_batches / total_batches * 100) if total_batches > 0 else 100.0
        priority_list = get_priority_list(total_loss)
        priority_df = pd.DataFrame(priority_list)
        if not priority_df.empty:
            priority_df = priority_df.sort_values("priority_score", ascending=False)
        suggestions = generate_suggestions(priority_list, total_loss, compliance_rate)
        suggestions_df = pd.DataFrame([{"建议标题": s["title"], "建议内容": s["content"]} for s in suggestions])
        conclusion_df = pd.DataFrame([
            {"指标名称": "总质量损失(元)", "指标值": round(total_loss, 2)},
            {"指标名称": "国标合规率", "指标值": f"{round(compliance_rate, 2)}%"},
            {"指标名称": "极高优先级改进项数量",
             "指标值": len(priority_df[priority_df["priority_level"] == "极高优先级"]) if not priority_df.empty else 0},
            {"指标名称": "高优先级改进项数量",
             "指标值": len(priority_df[priority_df["priority_level"] == "高优先级"]) if not priority_df.empty else 0},
            {"指标名称": "报告生成时间", "指标值": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ])
        output = export_excel(priority_df, "改进优先级与建议")
        import io
        with pd.ExcelWriter(output, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            suggestions_df.to_excel(writer, sheet_name="改进建议", index=False)
            conclusion_df.to_excel(writer, sheet_name="综合结论", index=False)
        output.seek(0)
        return send_file(
            output,
            download_name=f"改进方案报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            as_attachment=True
        )
    except Exception as e:
        return f"导出失败：{str(e)}", 500