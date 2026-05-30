#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成数据配置文件脚本
运行方式：python generate_json_files.py
功能：在项目根目录下创建 data 文件夹，并生成 default_ctqs.json 和 factor_library.json
"""

import json
from pathlib import Path

# ========== 1. 默认 CTQ 配置数据 ==========
DEFAULT_CTQS = [
    {
        "ctq_name": "蛋白质含量",
        "feature_type": "nominal",
        "process_link": "原料标准化",
        "is_ccp": "是",
        "fmea_severity": 6,
        "gb_code": "GB 19302-2010",
        "target_m": 3.1,
        "usl": 3.5,
        "lsl": 2.9,
        "delta0": 0.6,
        "delta": 0.2,
        "a0": 3000,
        "a": 800,
        "asymmetric_loss": "否",
        "k_upper": 0,
        "k_lower": 0,
        "a_upper": 0,
        "a_lower": 0,
        "hidden_loss_coef": 1.2,
        "status": "启用"
    },
    {
        "ctq_name": "滴定酸度",
        "feature_type": "nominal",
        "process_link": "发酵环节",
        "is_ccp": "是",
        "fmea_severity": 5,
        "gb_code": "GB 19302-2010",
        "target_m": 75,
        "usl": 85,
        "lsl": 70,
        "delta0": 15,
        "delta": 5,
        "a0": 2500,
        "a": 600,
        "asymmetric_loss": "否",
        "k_upper": 0,
        "k_lower": 0,
        "a_upper": 0,
        "a_lower": 0,
        "hidden_loss_coef": 1.1,
        "status": "启用"
    },
    {
        "ctq_name": "灌装净含量",
        "feature_type": "nominal",
        "process_link": "灌装环节",
        "is_ccp": "否",
        "fmea_severity": 4,
        "gb_code": "JJF 1070",
        "target_m": 200,
        "usl": 204.5,
        "lsl": 195.5,
        "delta0": 9,
        "delta": 4.5,
        "a0": 1.2,
        "a": 0.3,
        "asymmetric_loss": "是",
        "k_upper": 0.0037,
        "k_lower": 0.0148,
        "a_upper": 0.6,
        "a_lower": 1.2,
        "hidden_loss_coef": 1.0,
        "status": "启用"
    },
    {
        "ctq_name": "菌落总数",
        "feature_type": "smaller",
        "process_link": "成品检验",
        "is_ccp": "是",
        "fmea_severity": 10,
        "gb_code": "GB 19302-2010",
        "target_m": 0,
        "usl": 100,
        "lsl": 0,
        "delta0": 100,
        "delta": 50,
        "a0": 50000,
        "a": 50000,
        "asymmetric_loss": "否",
        "k_upper": 0,
        "k_lower": 0,
        "a_upper": 0,
        "a_lower": 0,
        "hidden_loss_coef": 5.0,
        "status": "启用"
    },
    {
        "ctq_name": "乳清析出率",
        "feature_type": "smaller",
        "process_link": "发酵环节",
        "is_ccp": "否",
        "fmea_severity": 6,
        "gb_code": "内控标准",
        "target_m": 0,
        "usl": 5,
        "lsl": 0,
        "delta0": 5,
        "delta": 2,
        "a0": 2500,
        "a": 700,
        "asymmetric_loss": "否",
        "k_upper": 0,
        "k_lower": 0,
        "a_upper": 0,
        "a_lower": 0,
        "hidden_loss_coef": 1.3,
        "status": "启用"
    },
    {
        "ctq_name": "保质期终点活菌数",
        "feature_type": "larger",
        "process_link": "仓储物流",
        "is_ccp": "是",
        "fmea_severity": 8,
        "gb_code": "GB 19302-2010",
        "target_m": 10000000,
        "usl": 1000000000,
        "lsl": 1000000,
        "delta0": 9000000,
        "delta": 5000000,
        "a0": 3000,
        "a": 0,
        "asymmetric_loss": "否",
        "k_upper": 0,
        "k_lower": 0,
        "a_upper": 0,
        "a_lower": 0,
        "hidden_loss_coef": 1.5,
        "status": "启用"
    },
    {
        "ctq_name": "冷链运输温度",
        "feature_type": "smaller",
        "process_link": "仓储物流",
        "is_ccp": "是",
        "fmea_severity": 9,
        "gb_code": "GB 14881-2013",
        "target_m": 2,
        "usl": 6,
        "lsl": 0,
        "delta0": 4,
        "delta": 2,
        "a0": 8000,
        "a": 1200,
        "asymmetric_loss": "否",
        "k_upper": 0,
        "k_lower": 0,
        "a_upper": 0,
        "a_lower": 0,
        "hidden_loss_coef": 1.8,
        "status": "启用"
    }
]

# ========== 2. 因子库数据 ==========
FACTOR_LIBRARY = {
    "双歧杆菌活菌数": {
        "features": [
            {"name": "初始添加量_log", "type": "numeric", "mean": 7.0, "std": 0.1, "desc": "log10(初始添加量 CFU/g)"},
            {"name": "共生菌比例", "type": "numeric", "mean": 2.0, "std": 0.3, "desc": "球菌:杆菌比例"},
            {"name": "发酵温度", "type": "numeric", "mean": 38.5, "std": 1.0, "desc": "℃"},
            {"name": "发酵时间", "type": "numeric", "mean": 8.0, "std": 1.0, "desc": "小时"},
            {"name": "终点pH", "type": "numeric", "mean": 4.5, "std": 0.1, "desc": ""},
            {"name": "总固形物", "type": "numeric", "mean": 13.5, "std": 1.0, "desc": "%"},
            {"name": "氧气含量", "type": "numeric", "mean": 0.2, "std": 0.1, "desc": "ppm"},
            {"name": "促进因子", "type": "categorical", "categories": ["FOS", "无"], "desc": "低聚果糖"},
            {"name": "透氧率", "type": "categorical", "categories": ["高阻隔", "普通"], "desc": "包装透氧率"},
            {"name": "贮藏温度", "type": "numeric", "mean": 4.0, "std": 0.5, "desc": "℃"},
            {"name": "贮藏天数", "type": "numeric", "mean": 10, "std": 7, "desc": "天"},
            {"name": "菌株", "type": "categorical", "categories": ["BB-12", "常规"], "desc": "抗逆性"}
        ],
        "formula_template": "10 ** (7.0 + 0.15*(初始添加量_log-7) + 0.08*(共生菌比例-2.0) + 0.20*(发酵温度-38.5)/1.0 - 0.15*(发酵时间-8)/1.0 - 0.40*(终点pH-4.5)/0.1 + 0.10*(总固形物-13.5)/1.0 - 0.05*(氧气含量-0.2)/0.1 + 0.30*(1 if 促进因子=='FOS' else 0) + 0.20*(1 if 透氧率=='高阻隔' else 0) - 0.05*(贮藏温度-4)/0.5 - 0.10*(贮藏天数/21) + 0.25*(1 if 菌株=='BB-12' else 0))"
    },
    "净含量(200g)": {
        "features": [
            {"name": "灌装机精度", "type": "numeric", "mean": 0.0, "std": 0.8, "desc": "g"},
            {"name": "产品温度", "type": "numeric", "mean": 8.0, "std": 1.0, "desc": "℃"},
            {"name": "产品粘度", "type": "numeric", "mean": 500, "std": 100, "desc": "cP", "distribution": "lognormal"},
            {"name": "灌装速度", "type": "numeric", "mean": 200, "std": 20, "desc": "瓶/分钟"},
            {"name": "容器重量波动", "type": "numeric", "mean": 0.0, "std": 1.0, "desc": "g"},
            {"name": "管道压力波动", "type": "numeric", "mean": 0.0, "std": 0.3, "desc": "bar"}
        ],
        "formula_template": "200.0 + 灌装机精度 + (产品温度-8)*0.2 - (log10(产品粘度)-log10(500))*2.0 - (灌装速度-200)/100 + 容器重量波动 + 管道压力波动*1.5"
    },
    "蛋白质含量": {
        "features": [
            {"name": "原料奶蛋白", "type": "numeric", "mean": 3.1, "std": 0.15, "desc": "%"},
            {"name": "均质压力", "type": "numeric", "mean": 20, "std": 2, "desc": "MPa"},
            {"name": "UHT温度", "type": "numeric", "mean": 140, "std": 1.5, "desc": "℃"},
            {"name": "储存时间", "type": "numeric", "mean": 90, "std": 60, "desc": "天"}
        ],
        "formula_template": "3.1 + 0.8*(原料奶蛋白-3.1) - 0.05*(均质压力-20)/2 - 0.10*(UHT温度-140)/1.5 - 0.05*(储存时间/180)"
    },
    "脂肪含量": {
        "features": [
            {"name": "原料奶脂肪", "type": "numeric", "mean": 3.8, "std": 0.20, "desc": "%"},
            {"name": "标准化分离效率", "type": "numeric", "mean": 0.98, "std": 0.01, "desc": ""},
            {"name": "均质压力", "type": "numeric", "mean": 20, "std": 2, "desc": "MPa"},
            {"name": "储存时间", "type": "numeric", "mean": 90, "std": 60, "desc": "天"}
        ],
        "formula_template": "3.5 + 0.9*(原料奶脂肪-3.8) - (1-标准化分离效率)*3.8 + 0.05*(均质压力-20)/2 - 0.02*(储存时间/180)"
    }
}


def main():
    # 创建 data 目录（如果不存在）
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # 写入 default_ctqs.json
    default_ctqs_path = data_dir / "default_ctqs.json"
    with open(default_ctqs_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CTQS, f, indent=2, ensure_ascii=False)
    print(f"✅ 已生成 {default_ctqs_path}")

    # 写入 factor_library.json
    factor_lib_path = data_dir / "factor_library.json"
    with open(factor_lib_path, "w", encoding="utf-8") as f:
        json.dump(FACTOR_LIBRARY, f, indent=2, ensure_ascii=False)
    print(f"✅ 已生成 {factor_lib_path}")

    print("\n🎉 JSON 配置文件生成完成！")
    print("提示：请将此脚本放在项目根目录（与 app.py 同级）并运行。")


if __name__ == "__main__":
    main()