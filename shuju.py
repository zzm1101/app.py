import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 设置随机种子确保结果可复现
np.random.seed(42)
random.seed(42)

# ---------------------- 批号规则配置 ----------------------
# 起始日期、总天数、每日批次数量
start_date = datetime(2026, 5, 1)
total_days = 25
batches_per_day = 20
total_batches = total_days * batches_per_day  # 总计500个批次

# 生成500个符合规则的批次号
batch_numbers = []
for day_offset in range(total_days):
    # 计算当前日期
    current_date = start_date + timedelta(days=day_offset)
    date_str = current_date.strftime("%Y%m%d")
    # 生成当日01-20的批次序号
    for batch_idx in range(1, batches_per_day + 1):
        batch_str = f"{batch_idx:02d}"
        full_batch_no = f"YOG{date_str}{batch_str}"
        batch_numbers.append(full_batch_no)
# -----------------------------------------------------------

# 影响因子定义（与原规则完全一致）
features = [
    ("发酵时间", "numeric", 7.0, 0.5, 5.5, 8.5),
    ("发酵温度", "numeric", 40.0, 0.8, 38.0, 42.0),
    ("菌种类型", "categorical", None, None, None, None),
    ("待装罐储存温度", "numeric", 4.0, 1.0, 2.0, 8.0),
    ("配料糖添加量", "numeric", 8.0, 0.7, 6.0, 10.0),
    ("配料蛋白质含量", "numeric", 3.5, 0.2, 3.0, 4.2)
]

# 菌种类型分布（与原规则完全一致）
strain_distribution = ["Bb-12"] * 200 + ["HN019"] * 150 + ["BL21"] * 100 + ["BB536"] * 50
random.shuffle(strain_distribution)

# 生成数据集
data = []
for batch_idx, batch_no in enumerate(batch_numbers):
    # 生成该批次的所有特征值
    feature_values = {}

    # 数值型特征：正态分布+截断（与原规则完全一致）
    for name, ftype, mean, std, min_val, max_val in features:
        if ftype == "numeric":
            value = np.random.normal(mean, std)
            value = max(min_val, min(max_val, value))
            feature_values[name] = round(value, 1)

    # 类别型特征：菌种类型
    feature_values["菌种类型"] = strain_distribution[batch_idx]

    # 计算target_value（双歧杆菌数）基于行业经验公式（与原规则完全一致）
    base = 1e8
    # 发酵温度影响：39-41℃最优
    temp = feature_values["发酵温度"]
    temp_factor = 1 - 0.1 * abs(temp - 40)
    # 发酵时间影响：6.5-7.5小时最优
    time = feature_values["发酵时间"]
    time_factor = 1 - 0.08 * abs(time - 7.0)
    # 蛋白质含量影响：越高越好
    protein = feature_values["配料蛋白质含量"]
    protein_factor = 0.8 + 0.2 * (protein - 3.0) / 1.2
    # 储存温度影响：越低越好
    storage = feature_values["待装罐储存温度"]
    storage_factor = 1 - 0.05 * (storage - 2.0)
    # 糖含量影响：8%最优
    sugar = feature_values["配料糖添加量"]
    sugar_factor = 1 - 0.03 * abs(sugar - 8.0)
    # 菌种类型影响
    strain = feature_values["菌种类型"]
    strain_factors = {"Bb-12": 1.0, "HN019": 0.95, "BL21": 0.88, "BB536": 0.82}
    strain_factor = strain_factors[strain]
    # 综合计算
    target = base * temp_factor * time_factor * protein_factor * storage_factor * sugar_factor * strain_factor
    target = round(target, 1)

    # 添加6条记录（每个特征一条）
    for name, ftype, _, _, _, _ in features:
        data.append({
            "batch_no": batch_no,
            "ctq_name": "双歧杆菌数",
            "feature_name": name,
            "feature_value": feature_values[name],
            "feature_type": ftype,
            "target_value": target
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 保存为CSV文件
df.to_csv("酸奶双歧杆菌影响因子数据集_500批次_日期批号版.csv", index=False, encoding="utf-8-sig")

# 输出验证信息
print("✅ 数据集生成完成！")
print(f"📊 总记录数：{len(df)}条")
print(f"📦 批次数量：{len(batch_numbers)}个")
print(
    f"📅 日期范围：{start_date.strftime('%Y-%m-%d')} 至 {(start_date + timedelta(days=total_days - 1)).strftime('%Y-%m-%d')}")
print(f"🔢 每日批次：{batches_per_day}个")
print(f"💾 文件已保存为：酸奶双歧杆菌影响因子数据集_500批次_日期批号版.csv")
print("\n📌 批号示例：")
print(f"  2026-05-01 第01批次：{batch_numbers[0]}")
print(f"  2026-05-01 第20批次：{batch_numbers[19]}")
print(f"  2026-05-02 第01批次：{batch_numbers[20]}")
print(f"  2026-05-25 第20批次：{batch_numbers[-1]}")