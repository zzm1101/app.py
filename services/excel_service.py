# services/excel_service.py
# Excel 导入导出辅助服务

import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from models.models import CTQConfig
from config import PRODUCT_ITEMS, FEATURE_TYPE

def generate_production_template():
    template_df = pd.DataFrame([
        {
            "生产批次号": "YOG2024050101",
            "生产日期": "2024-05-01",
            "生产线": "1号线",
            "生产班次": "早班",
            "品项": "原味",
            "样品编号": "S2024050101-1",
            "CTQ编号": 1,
            "CTQ名称": "蛋白质含量",
            "实测值": 3.1,
            "批次生产数量": 10000,
            "存储天数": 0,
            "存储温度(℃)": 4,
            "检验员": "张三"
        }
    ])
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, sheet_name="生产数据模板", index=False)
    output.seek(0)
    return output

def generate_ctq_template():
    sample_data = []
    default_ctqs = [
        {"ctq_name": "蛋白质含量", "feature_type": "nominal", "process_link": "原料标准化", "is_ccp": "是",
         "fmea_severity": 6, "gb_code": "GB 19302-2010", "target_m": 3.1, "usl": 3.5, "lsl": 2.9,
         "delta0": 0.6, "delta": 0.2, "a0": 3000, "a": 800, "asymmetric_loss": "否", "k_upper": 0, "k_lower": 0,
         "a_upper": 0, "a_lower": 0, "hidden_loss_coef": 1.2, "status": "启用"},
        {"ctq_name": "灌装净含量", "feature_type": "nominal", "process_link": "灌装环节", "is_ccp": "否",
         "fmea_severity": 4, "gb_code": "JJF 1070", "target_m": 200, "usl": 204.5, "lsl": 195.5,
         "delta0": 9, "delta": 4.5, "a0": 1.2, "a": 0.3, "asymmetric_loss": "是", "k_upper": 0.0037, "k_lower": 0.0148,
         "a_upper": 0.6, "a_lower": 1.2, "hidden_loss_coef": 1.0, "status": "启用"},
    ]
    for ctq in default_ctqs:
        row = {"品项": "", **ctq}
        sample_data.append(row)
        for item in PRODUCT_ITEMS[:2]:
            row_item = {"品项": item, **ctq}
            sample_data.append(row_item)
    df = pd.DataFrame(sample_data)
    columns_order = ["品项", "ctq_name", "feature_type", "process_link", "is_ccp", "fmea_severity",
                     "gb_code", "target_m", "usl", "lsl", "delta0", "delta", "a0", "a",
                     "asymmetric_loss", "k_upper", "k_lower", "a_upper", "a_lower",
                     "hidden_loss_coef", "status"]
    df = df[columns_order]
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        desc_df = pd.DataFrame({
            "说明": [
                "1. 品项列：留空表示通用配置；填写具体品项名称为专属配置。",
                "2. 必填字段：ctq_name, feature_type, usl, lsl, target_m",
                "3. feature_type 必须为：nominal(望目), smaller(望小), larger(望大)",
                "4. 品项名称请从系统允许列表中选择：" + ", ".join(PRODUCT_ITEMS),
                "5. 上传时将自动覆盖已存在的(品项+CTQ名称)组合。",
                '6. 非对称损失为"是"时，必须填写 k_upper 和 k_lower。',
                "7. a_upper 和 a_lower 为非对称损失时的上下限损失金额（元）。"
            ]
        })
        desc_df.to_excel(writer, sheet_name="填写说明", index=False)
        df.to_excel(writer, sheet_name="数据模板", index=False)
    output.seek(0)
    return output

def generate_mock_production_data():
    np.random.seed(42)
    start_date = datetime(2024, 4, 1)
    all_ctq = CTQConfig.query.filter_by(status="启用").all()
    if not all_ctq:
        return pd.DataFrame()
    ctq_map = {}
    for ctq in all_ctq:
        item = ctq.product_item if ctq.product_item else '__global__'
        ctq_map[(item, ctq.ctq_name.strip())] = ctq
    def find_ctq(item_name, ctq_name):
        ctq_name = ctq_name.strip()
        if (item_name, ctq_name) in ctq_map:
            return ctq_map[(item_name, ctq_name)]
        if ('__global__', ctq_name) in ctq_map:
            return ctq_map[('__global__', ctq_name)]
        return None
    db_items = CTQConfig.query.with_entities(CTQConfig.product_item).filter(
        CTQConfig.product_item != None,
        CTQConfig.status == "启用"
    ).distinct().all()
    distinct_items = [item[0] for item in db_items if item[0]]
    if not distinct_items:
        distinct_items = PRODUCT_ITEMS
    if not distinct_items:
        distinct_items = ["示例品项"]
    items = distinct_items
    ctq_names = list(set(ctq.ctq_name for ctq in all_ctq))
    SAMPLES_PER_BATCH = 5
    batch_list = []
    for batch_idx in range(60):
        produce_date = start_date + timedelta(days=batch_idx // 3)
        batch_no = f"YOG{produce_date.strftime('%Y%m%d')}{batch_idx % 3 + 1:02d}"
        product_line = f"{(batch_idx % 2) + 1}号线"
        work_shift = ["早班", "中班", "晚班"][batch_idx % 3]
        product_item = items[batch_idx % len(items)]
        batch_quantity = np.random.randint(8000, 12000)
        for ctq_name in ctq_names:
            ctq = find_ctq(product_item, ctq_name)
            if not ctq:
                continue
            mean_val, std_val = get_smart_mean_std(ctq)
            base_measured = np.random.normal(mean_val, std_val)
            if ctq.lsl and ctq.usl:
                base_measured = np.clip(base_measured, ctq.lsl * 0.9, ctq.usl * 1.1)
            for sample_idx in range(SAMPLES_PER_BATCH):
                noise_ratio = np.random.uniform(-0.02, 0.02)
                sample_val = base_measured * (1 + noise_ratio)
                if ctq.lsl and ctq.usl:
                    sample_val = np.clip(sample_val, ctq.lsl * 0.85, ctq.usl * 1.15)
                if ctq.feature_type == "larger" and sample_val <= 0:
                    sample_val = max(ctq.lsl if ctq.lsl else 1, 1e6)
                if ctq_name == "保质期终点活菌数" or abs(mean_val) > 1e6:
                    formatted_val = round(sample_val, 0)
                else:
                    formatted_val = round(sample_val, 4)
                batch_list.append({
                    "生产批次号": batch_no,
                    "生产日期": produce_date.date(),
                    "生产线": product_line,
                    "生产班次": work_shift,
                    "品项": product_item,
                    "样品编号": f"S{batch_no}-{sample_idx+1}",
                    "CTQ编号": ctq.ctq_id,
                    "CTQ名称": ctq.ctq_name,
                    "实测值": formatted_val,
                    "批次生产数量": batch_quantity,
                    "存储天数": 0,
                    "存储温度(℃)": 4,
                    "检验员": ["张三", "李四", "王五"][batch_idx % 3]
                })
    return pd.DataFrame(batch_list)

def get_smart_mean_std(ctq):
    known_ctqs = {
        "蛋白质含量": (3.1, 0.08),
        "滴定酸度": (75, 1.5),
        "灌装净含量": (200, 1.2),
        "菌落总数": (10, 5),
        "乳清析出率": (2.5, 0.8),
        "保质期终点活菌数": (5e7, 1e7),
        "冷链运输温度": (4.5, 1.0),
    }
    if ctq.ctq_name in known_ctqs:
        return known_ctqs[ctq.ctq_name]
    usl = ctq.usl
    lsl = ctq.lsl
    target = ctq.target_m if ctq.target_m and ctq.target_m != 0 else (usl + lsl) / 2
    if usl and lsl and usl > lsl:
        std = (usl - lsl) / 10.0
    else:
        std = 1.0
    if ctq.feature_type == "smaller":
        mean = lsl + (usl - lsl) * 0.1 if lsl and usl else target
    elif ctq.feature_type == "larger":
        mean = usl - (usl - lsl) * 0.1 if lsl and usl else target
    else:
        mean = target
    return mean, std

def export_excel(df, sheet_name):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output