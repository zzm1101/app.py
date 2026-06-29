#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
密封磨损监测系统 - 模拟数据生成器
生成符合系统要求的 CSV 格式数据，包含退化趋势
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_healthy_data(days=30, interval_minutes=10, seed=42):
    """
    生成健康设备的模拟数据

    参数:
        days: 数据天数
        interval_minutes: 采样间隔（分钟）
        seed: 随机种子

    返回:
        DataFrame
    """
    np.random.seed(seed)

    # 计算总点数
    points_per_day = 24 * 60 // interval_minutes
    total_points = days * points_per_day

    # 生成时间戳
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_points)]

    # 基础参数（正常工况）
    base_params = {
        'ae_energy': 5.0,  # 声发射能量基准值
        'ae_count': 100,  # 声发射计数基准值
        'vibration_rms': 1.0,  # 振动RMS基准值
        'temperature': 40,  # 温度基准值(°C)
        'current': 12,  # 电流基准值(A)
        'pressure': 3.0,  # 压力基准值(bar)
    }

    # 噪声水平（健康设备噪声较小）
    noise_level = {
        'ae_energy': 0.3,
        'ae_count': 8,
        'vibration_rms': 0.08,
        'temperature': 1.5,
        'current': 0.2,
        'pressure': 0.08,
    }

    data = []
    for i, ts in enumerate(timestamps):
        row = {'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S')}

        for param, base in base_params.items():
            # 添加随机噪声
            noise = np.random.normal(0, noise_level[param])
            value = base + noise
            # 确保合理范围
            if param == 'ae_energy':
                value = max(0.1, min(15, value))
            elif param == 'ae_count':
                value = max(10, min(200, value))
            elif param == 'vibration_rms':
                value = max(0.1, min(3.0, value))
            elif param == 'temperature':
                value = max(30, min(55, value))
            elif param == 'current':
                value = max(8, min(18, value))
            elif param == 'pressure':
                value = max(2.0, min(4.5, value))

            row[param] = round(value, 4)

        data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_degrading_data(days=90, interval_minutes=10, seed=42,
                            degradation_start_day=30, degradation_rate=0.005):
    """
    生成具有退化趋势的模拟数据（用于RUL预测）

    参数:
        days: 数据天数
        interval_minutes: 采样间隔（分钟）
        seed: 随机种子
        degradation_start_day: 开始退化的天数
        degradation_rate: 退化速率（每天健康度下降）

    返回:
        DataFrame
    """
    np.random.seed(seed)

    points_per_day = 24 * 60 // interval_minutes
    total_points = days * points_per_day

    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_points)]

    # 基础参数（初始健康状态）
    base_params = {
        'ae_energy': 4.5,
        'ae_count': 90,
        'vibration_rms': 0.9,
        'temperature': 39.5,
        'current': 11.8,
        'pressure': 2.95,
    }

    # 健康设备噪声水平
    healthy_noise = {
        'ae_energy': 0.25,
        'ae_count': 6,
        'vibration_rms': 0.06,
        'temperature': 1.2,
        'current': 0.15,
        'pressure': 0.06,
    }

    # 退化后噪声增加
    degraded_noise_multiplier = {
        'ae_energy': 2.5,
        'ae_count': 2.0,
        'vibration_rms': 3.0,
        'temperature': 1.5,
        'current': 1.8,
        'pressure': 1.5,
    }

    data = []
    for i, ts in enumerate(timestamps):
        # 计算当前天数和健康度
        current_day = i / points_per_day
        health = 1.0

        if current_day > degradation_start_day:
            degradation_progress = (current_day - degradation_start_day) / (days - degradation_start_day)
            health = 1.0 - degradation_progress * degradation_rate * days
            health = max(0.2, health)

        # 退化影响因子（健康度越低，参数偏离越大）
        degradation_factor = 1.0 - health
        # 噪声放大因子
        noise_factor = 1.0 + degradation_factor * 2

        row = {'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S')}

        for param, base in base_params.items():
            # 计算退化偏移
            if param in ['ae_energy', 'ae_count', 'vibration_rms', 'temperature']:
                # 这些参数随退化增加
                offset = degradation_factor * base * 0.3
                value = base + offset
            elif param in ['current', 'pressure']:
                # 电流和压力随退化变化
                offset = degradation_factor * base * 0.15
                value = base + offset * (1 if param == 'current' else -1)
            else:
                value = base

            # 添加噪声（退化后噪声增大）
            noise_base = healthy_noise.get(param, 0.1)
            noise = np.random.normal(0, noise_base * noise_factor)
            value += noise

            # 确保合理范围
            if param == 'ae_energy':
                value = max(0.1, min(20, value))
            elif param == 'ae_count':
                value = max(10, min(300, value))
            elif param == 'vibration_rms':
                value = max(0.1, min(5.0, value))
            elif param == 'temperature':
                value = max(30, min(65, value))
            elif param == 'current':
                value = max(8, min(22, value))
            elif param == 'pressure':
                value = max(2.0, min(5.0, value))

            row[param] = round(value, 4)

        data.append(row)

    df = pd.DataFrame(data)
    return df


def add_cip_cycles(df, cip_interval_hours=24, cip_duration_minutes=30):
    """
    在数据中添加 CIP 清洗周期（温度升高）

    参数:
        df: 原始数据DataFrame
        cip_interval_hours: CIP 间隔（小时）
        cip_duration_minutes: CIP 持续时间（分钟）

    返回:
        修改后的DataFrame
    """
    df = df.copy()
    timestamps = pd.to_datetime(df['timestamp'])

    # 计算 CIP 开始时间点
    start_time = timestamps.min()
    cip_interval_seconds = cip_interval_hours * 3600
    cip_duration_seconds = cip_duration_minutes * 60

    for i, ts in enumerate(timestamps):
        seconds_since_start = (ts - start_time).total_seconds()

        # 检查是否在 CIP 周期内
        cip_cycle_num = int(seconds_since_start // cip_interval_seconds)
        cip_start = start_time + timedelta(seconds=cip_cycle_num * cip_interval_seconds)
        cip_end = cip_start + timedelta(seconds=cip_duration_seconds)

        if cip_start <= ts < cip_end:
            # CIP 期间温度升高
            df.loc[i, 'temperature'] = min(85, df.loc[i, 'temperature'] + 35)
            df.loc[i, 'current'] = min(15, df.loc[i, 'current'] + 2)

    return df


def generate_test_dataset():
    """生成完整的测试数据集"""

    print("=" * 60)
    print("密封磨损监测系统 - 模拟数据生成器")
    print("=" * 60)

    # 创建输出目录
    output_dir = "seal_data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成健康设备数据（30天，10分钟间隔）
    print("\n[1/4] 生成健康设备数据...")
    df_healthy = generate_healthy_data(days=30, interval_minutes=10)
    df_healthy_with_cip = add_cip_cycles(df_healthy, cip_interval_hours=24)

    healthy_path = os.path.join(output_dir, "device_healthy_30days.csv")
    df_healthy_with_cip.to_csv(healthy_path, index=False)
    print(f"  ✅ 已生成: {healthy_path}")
    print(f"     行数: {len(df_healthy_with_cip)}")

    # 2. 生成退化设备数据（90天，10分钟间隔）
    print("\n[2/4] 生成退化设备数据（用于RUL预测）...")
    df_degrading = generate_degrading_data(
        days=90,
        interval_minutes=10,
        degradation_start_day=30,
        degradation_rate=0.008
    )
    df_degrading_with_cip = add_cip_cycles(df_degrading, cip_interval_hours=24)

    degrading_path = os.path.join(output_dir, "device_degrading_90days.csv")
    df_degrading_with_cip.to_csv(degrading_path, index=False)
    print(f"  ✅ 已生成: {degrading_path}")
    print(f"     行数: {len(df_degrading_with_cip)}")

    # 3. 生成短期测试数据（7天，1分钟间隔，用于快速测试）
    print("\n[3/4] 生成短期测试数据（7天，1分钟间隔）...")
    df_short = generate_degrading_data(
        days=7,
        interval_minutes=1,
        degradation_start_day=3,
        degradation_rate=0.01
    )
    df_short_with_cip = add_cip_cycles(df_short, cip_interval_hours=12, cip_duration_minutes=20)

    short_path = os.path.join(output_dir, "device_test_7days.csv")
    df_short_with_cip.to_csv(short_path, index=False)
    print(f"  ✅ 已生成: {short_path}")
    print(f"     行数: {len(df_short_with_cip)}")

    # 4. 生成非常小的测试数据（500行，用于快速验证）
    print("\n[4/4] 生成快速验证数据（500行）...")
    df_quick = generate_degrading_data(
        days=3,
        interval_minutes=8,
        degradation_start_day=1,
        degradation_rate=0.02
    )

    quick_path = os.path.join(output_dir, "device_quick_test.csv")
    df_quick.to_csv(quick_path, index=False)
    print(f"  ✅ 已生成: {quick_path}")
    print(f"     行数: {len(df_quick)}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据生成完成！统计信息：")
    print("=" * 60)

    for path in [healthy_path, degrading_path, short_path, quick_path]:
        df = pd.read_csv(path)
        print(f"\n📁 {os.path.basename(path)}")
        print(f"   行数: {len(df)}")
        print(f"   温度范围: {df['temperature'].min():.1f} ~ {df['temperature'].max():.1f}°C")
        print(f"   电流范围: {df['current'].min():.1f} ~ {df['current'].max():.1f}A")
        print(f"   时间范围: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

    print("\n" + "=" * 60)
    print("使用说明:")
    print("1. 访问 http://127.0.0.1:5001/seal")
    print("2. 在「配置」标签页添加设备（如「测试设备01」）")
    print("3. 切换到「上传」标签页")
    print("4. 选择设备，上传生成的 CSV 文件")
    print("5. 切换到设备页面，点击「分析」按钮")
    print("=" * 60)

    return {
        'healthy': healthy_path,
        'degrading': degrading_path,
        'test': short_path,
        'quick': quick_path
    }


def generate_single_file_for_upload():
    """
    生成单个可直接上传的 CSV 文件（放在项目根目录）
    """
    print("生成可直接上传的 CSV 文件...")

    # 生成数据
    df = generate_degrading_data(
        days=14,
        interval_minutes=5,
        degradation_start_day=5,
        degradation_rate=0.006
    )

    # 添加 CIP 周期
    df = add_cip_cycles(df, cip_interval_hours=12, cip_duration_minutes=30)

    # 保存到根目录
    output_path = "upload_ready.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ 已生成可直接上传的文件: {output_path}")
    print(f"   行数: {len(df)}")
    print(f"   列名: {list(df.columns)}")
    print(f"   温度范围: {df['temperature'].min():.1f} ~ {df['temperature'].max():.1f}°C")
    print("\n请在密封监测系统中上传此文件")

    return output_path


if __name__ == "__main__":
    # 生成完整测试数据集
    files = generate_test_dataset()

    # 同时生成一个可直接上传的单文件
    print("\n")
    generate_single_file_for_upload()