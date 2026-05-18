# services/spc_service.py
# SPC控制图计算与能力分析核心服务

import numpy as np
from collections import defaultdict
from scipy.stats import norm, shapiro, anderson
from datetime import datetime
from models.models import LossResult, CTQConfig, db
from sqlalchemy import func

# ========== 新增常数计算函数（支持任意子组大小） ==========
def get_d2(n):
    """子组大小为 n 时的 d2 常数（极差估计标准差）"""
    if n == 1:
        return 1.0
    # 精确值查表（n<=25）
    d2_table = {
        2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
        8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336,
        14: 3.407, 15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689,
        20: 3.735, 21: 3.778, 22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931
    }
    if n <= 25:
        return d2_table.get(n, 3.931)
    else:
        # 大 n 近似公式
        from scipy.special import gamma
        return 2 * np.sqrt(2/(n-1)) * gamma(n/2) / gamma((n-1)/2)


def get_c4(n):
    """子组大小为 n 时的 c4 常数（标准差无偏估计）"""
    c4_table = {
        2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594,
        8: 0.9650, 9: 0.9693, 10: 0.9727, 11: 0.9754, 12: 0.9776, 13: 0.9794,
        14: 0.9810, 15: 0.9823, 16: 0.9835, 17: 0.9845, 18: 0.9854, 19: 0.9862,
        20: 0.9869, 21: 0.9876, 22: 0.9882, 23: 0.9887, 24: 0.9892, 25: 0.9896
    }
    if n <= 25:
        return c4_table.get(n, 0.9896)
    else:
        return 1 - 1/(4*n) - 3/(64*n**2)


def get_A2(n):
    """Xbar-R 图 A2 常数"""
    d2 = get_d2(n)
    return 3 / (d2 * np.sqrt(n))


def get_D3_D4(n):
    """Xbar-R 图 D3, D4 常数"""
    # d3 表（极差的标准差）
    d3_table = {
        2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820,
        9: 0.808, 10: 0.797, 11: 0.787, 12: 0.778, 13: 0.770, 14: 0.763, 15: 0.756
    }
    d3 = d3_table.get(n, 0.756)
    d2 = get_d2(n)
    D3 = max(0, 1 - 3 * d3 / d2)
    D4 = 1 + 3 * d3 / d2
    return D3, D4


def get_A3(n):
    """Xbar-S 图 A3 常数"""
    c4 = get_c4(n)
    return 3 / (c4 * np.sqrt(n))


def get_B3_B4(n):
    """Xbar-S 图 B3, B4 常数（S 控制限）"""
    c4 = get_c4(n)
    # 标准差的标准差近似
    sigma_s = c4 * np.sqrt(1 - c4**2)
    B3 = max(0, 1 - 3 * sigma_s / c4)
    B4 = 1 + 3 * sigma_s / c4
    return B3, B4


# ========== SPC 预警生成 ==========
def generate_spc_alerts():
    alerts = []
    groups = db.session.query(
        LossResult.product_item,
        LossResult.ctq_id,
        LossResult.ctq_name
    ).filter(LossResult.product_item != None) \
     .group_by(LossResult.product_item, LossResult.ctq_id).all()

    for item, ctq_id, ctq_name in groups:
        losses = db.session.query(LossResult.batch_total_loss).filter(
            LossResult.product_item == item,
            LossResult.ctq_id == ctq_id
        ).order_by(LossResult.calc_time.desc()).limit(6).all()
        if len(losses) < 3:
            continue
        loss_vals = [l[0] for l in losses]
        if all(loss_vals[i] < loss_vals[i+1] for i in range(len(loss_vals)-1)):
            alerts.append(f"【{item} - {ctq_name}】连续{len(loss_vals)}批损失递增，请核查过程稳定性")
        all_losses = db.session.query(LossResult.batch_total_loss).filter(
            LossResult.product_item == item,
            LossResult.ctq_id == ctq_id
        ).all()
        if len(all_losses) < 10:
            continue
        all_vals = np.array([v[0] for v in all_losses])
        mean_loss = np.mean(all_vals)
        std_loss = np.std(all_vals, ddof=1)
        ucl = mean_loss + 3*std_loss
        for bl in losses[:3]:
            if bl[0] > ucl:
                alerts.append(f"【{item} - {ctq_name}】批次损失 {bl[0]:.2f} 超出控制上限({ucl:.2f})")
                break
    return alerts


# ========== 控制图计算主函数 ==========
def compute_control_chart(production_data, ctq_usl=None, ctq_lsl=None, ctq_target=None,
                         chart_type='auto', rules_active=None, subgroup_size=None):
    if rules_active is None:
        rules_active = list(range(1, 9))

    # 自然分组：按批次号 batch_no
    batch_dict = defaultdict(list)
    batch_order = []
    for row in production_data:
        batch = row.batch_no
        batch_dict[batch].append(row.measured_value)
        if batch not in batch_order:
            batch_order.append((row.produce_date, batch))
    if not batch_dict:
        return {"error": "无数据"}
    batch_order.sort(key=lambda x: x[0])
    batch_labels = [b[1] for b in batch_order]
    batch_dates = [str(next((r.produce_date for r in production_data if r.batch_no == b), b)) for b in batch_labels]

    xbar, r_vals, s_vals = [], [], []
    all_values = []
    for b in batch_labels:
        vals = np.array(batch_dict[b])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        all_values.extend(vals)
        xbar.append(np.mean(vals))
        if len(vals) > 1:
            r_vals.append(np.max(vals) - np.min(vals))
            s_vals.append(np.std(vals, ddof=1))
        else:
            r_vals.append(None)
            s_vals.append(None)

    subgroup_sizes = [len(batch_dict[b]) for b in batch_labels]
    constant_n = len(set(subgroup_sizes)) == 1
    n = subgroup_sizes[0] if constant_n else None

    user_requested = chart_type
    message = None
    if chart_type == 'auto':
        if not constant_n or n is None or n == 1:
            chart_type = 'I-MR'
        elif n >= 10:
            chart_type = 'Xbar-S'
        else:
            chart_type = 'Xbar-R'
    else:
        if not constant_n or n is None or n < 2:
            if chart_type in ('Xbar-R', 'Xbar-S'):
                chart_type = 'I-MR'
                message = f"数据不适合所选类型（子组大小不恒定或n<2），已自动切换为 {chart_type}"
            else:
                message = None

    if chart_type == 'I-MR':
        mr = [abs(xbar[i] - xbar[i-1]) for i in range(1, len(xbar))]
        mr_bar = np.mean(mr) if mr else 0
        center_x = np.mean(xbar)
        d2 = get_d2(2)   # n=2 的 d2 = 1.128
        sigma_within_chart = mr_bar / d2
        ucl_x = center_x + 3 * mr_bar / d2
        lcl_x = center_x - 3 * mr_bar / d2
        ucl_r = 3.267 * mr_bar
        lcl_r = 0
        center_r = mr_bar
        plot_r = mr
        r_label = 'MR'
    elif chart_type == 'Xbar-R':
        r_clean = [v for v in r_vals if v is not None]
        r_bar = np.mean(r_clean) if r_clean else 0
        center_x = np.mean(xbar)
        d2 = get_d2(n)
        sigma_within_chart = r_bar / d2
        A2 = get_A2(n)
        D3, D4 = get_D3_D4(n)
        ucl_x = center_x + A2 * r_bar
        lcl_x = center_x - A2 * r_bar
        ucl_r = D4 * r_bar
        lcl_r = D3 * r_bar
        center_r = r_bar
        plot_r = [v if v is not None else 0 for v in r_vals]
        r_label = 'R'
    else:  # Xbar-S
        s_clean = [v for v in s_vals if v is not None]
        s_bar = np.mean(s_clean) if s_clean else 0
        center_x = np.mean(xbar)
        c4 = get_c4(n)
        sigma_within_chart = s_bar / c4
        A3 = get_A3(n)
        ucl_x = center_x + A3 * s_bar
        lcl_x = center_x - A3 * s_bar
        B3, B4 = get_B3_B4(n)
        ucl_r = B4 * s_bar
        lcl_r = B3 * s_bar
        center_r = s_bar
        plot_r = s_vals
        r_label = 'S'

    alarm_x, alarm_r = [], []
    point_rules = [set() for _ in range(len(xbar))]
    point_rules_r = [set() for _ in range(len(plot_r))]

    # 规则1：超出3σ控制限
    if 1 in rules_active:
        for i, v in enumerate(xbar):
            if v > ucl_x or v < lcl_x:
                point_rules[i].add(1)
        for i, v in enumerate(plot_r):
            if v is not None and (v > ucl_r or v < lcl_r):
                point_rules_r[i].add(1)

    # 规则2：连续9点同侧
    if 2 in rules_active:
        side, cnt, start = None, 0, 0
        for i, v in enumerate(xbar):
            new_side = 1 if v > center_x else (0 if v < center_x else -1)
            if new_side == side and new_side != -1:
                cnt += 1
            else:
                side, cnt, start = new_side, 1, i
            if cnt >= 9:
                for j in range(start, i+1):
                    point_rules[j].add(2)
                side, cnt = None, 0

    # 规则3：连续6点递增或递减
    if 3 in rules_active:
        for i in range(len(xbar)-5):
            if all(xbar[i+j] < xbar[i+j+1] for j in range(5)):
                for j in range(i, i+6):
                    point_rules[j].add(3)
            if all(xbar[i+j] > xbar[i+j+1] for j in range(5)):
                for j in range(i, i+6):
                    point_rules[j].add(3)

    # 规则4：连续14点交替上下
    if 4 in rules_active:
        for i in range(len(xbar)-13):
            if all((xbar[i+j]-xbar[i+j+1])*(xbar[i+j+1]-xbar[i+j+2]) < 0 for j in range(12)):
                for j in range(i, i+14):
                    point_rules[j].add(4)

    # 规则5：连续3点中有2点在2σ外（同侧）
    if 5 in rules_active:
        s2u = center_x + 2 * (ucl_x - center_x) / 3   # 近似 2σ 上界
        s2l = center_x - 2 * (center_x - lcl_x) / 3   # 近似 2σ 下界
        for i in range(len(xbar)-2):
            w = xbar[i:i+3]
            if sum(1 for v in w if v > s2u) >= 2:
                for j in range(i, i+3):
                    point_rules[j].add(5)
            if sum(1 for v in w if v < s2l) >= 2:
                for j in range(i, i+3):
                    point_rules[j].add(5)

    # 规则6：连续5点中有4点在1σ外（同侧）
    if 6 in rules_active:
        s1u = center_x + (ucl_x - center_x) / 3   # 近似 1σ 上界
        s1l = center_x - (center_x - lcl_x) / 3   # 近似 1σ 下界
        for i in range(len(xbar)-4):
            w = xbar[i:i+5]
            if sum(1 for v in w if v > s1u) >= 4:
                for j in range(i, i+5):
                    point_rules[j].add(6)
            if sum(1 for v in w if v < s1l) >= 4:
                for j in range(i, i+5):
                    point_rules[j].add(6)

    # 规则7：连续15点在1σ内
    if 7 in rules_active:
        s1u = center_x + (ucl_x - center_x) / 3
        s1l = center_x - (center_x - lcl_x) / 3
        for i in range(len(xbar)-14):
            if all(s1l <= xbar[i+j] <= s1u for j in range(15)):
                for j in range(i, i+15):
                    point_rules[j].add(7)

    # 规则8：连续8点在1σ外（两侧均可）
    if 8 in rules_active:
        s1u = center_x + (ucl_x - center_x) / 3
        s1l = center_x - (center_x - lcl_x) / 3
        for i in range(len(xbar)-7):
            if all(v > s1u or v < s1l for v in xbar[i:i+8]):
                for j in range(i, i+8):
                    point_rules[j].add(8)

    rules_violations = []
    for i, rs in enumerate(point_rules):
        if rs:
            alarm_x.append(i)
            rules_violations.append(f"批次{batch_labels[i]} 均值{xbar[i]:.4f} 违反规则{','.join(map(str,sorted(rs)))}")
    for i, rs in enumerate(point_rules_r):
        if rs:
            alarm_r.append(i)
            rules_violations.append(f"批次{batch_labels[i]} {r_label}{plot_r[i]:.4f} 超出控制限")

    all_arr = np.array(all_values)
    mean_total = np.mean(all_arr)
    std_total = np.std(all_arr, ddof=1) if len(all_arr) > 1 else 0

    all_subgroup_size_one = all(len(batch_dict[b]) == 1 for b in batch_labels)
    if all_subgroup_size_one:
        mr = [abs(xbar[i] - xbar[i-1]) for i in range(1, len(xbar))]
        mr_bar = np.mean(mr) if mr else 0
        sigma_within = mr_bar / 1.128
    else:
        pool_ss = 0.0
        pool_df = 0
        for b in batch_labels:
            vals = np.array(batch_dict[b])
            if len(vals) >= 2:
                pool_ss += (len(vals) - 1) * np.var(vals, ddof=1)
                pool_df += (len(vals) - 1)
        if pool_df > 0:
            sigma_within = np.sqrt(pool_ss / pool_df)
        else:
            sigma_within = 0.0

    if ctq_usl and ctq_lsl and sigma_within > 0:
        cpu = (ctq_usl - mean_total) / (3 * sigma_within)
        cpl = (mean_total - ctq_lsl) / (3 * sigma_within)
        cpk = min(cpu, cpl)
        cp = (ctq_usl - ctq_lsl) / (6 * sigma_within)
    else:
        cpk = None
        cp = None

    if ctq_usl and ctq_lsl and std_total > 0:
        cpu_overall = (ctq_usl - mean_total) / (3 * std_total)
        cpl_overall = (mean_total - ctq_lsl) / (3 * std_total)
        ppk = min(cpu_overall, cpl_overall)
        pp = (ctq_usl - ctq_lsl) / (6 * std_total)
    else:
        ppk = None
        pp = None

    cpm_val = None
    if ctq_usl and ctq_lsl and ctq_target and std_total > 0:
        bias = mean_total - ctq_target
        cpm_val = (ctq_usl - ctq_lsl) / (6 * np.sqrt(std_total**2 + bias**2))

    cpmk_val = None
    if ctq_usl and ctq_lsl and ctq_target and sigma_within > 0:
        sigma_cpm = np.sqrt(sigma_within**2 + (mean_total - ctq_target)**2)
        if sigma_cpm > 0:
            cpmk_upper = (ctq_usl - ctq_target) / (3 * sigma_cpm)
            cpmk_lower = (ctq_target - ctq_lsl) / (3 * sigma_cpm)
            cpmk_val = min(cpmk_upper, cpmk_lower)
        else:
            cpmk_val = None
    else:
        cpmk_val = None

    z_usl = (ctq_usl - mean_total) / std_total if ctq_usl and std_total > 0 else None
    z_lsl = (mean_total - ctq_lsl) / std_total if ctq_lsl and std_total > 0 else None
    z_bench = min(z_usl, z_lsl) if z_usl and z_lsl else None

    def ppm_from_z(z):
        if z is None: return None
        return 2 * (1 - norm.cdf(abs(z))) * 1e6 if z != 0 else 0

    ppm_total = ppm_from_z(z_bench) if z_bench else None
    ppm_usl = (1 - norm.cdf(z_usl)) * 1e6 if z_usl else None
    ppm_lsl = norm.cdf(-z_lsl) * 1e6 if z_lsl else None

    sw_stat, sw_p = None, None
    ad_result = None
    ad_pass = None
    qq_points = []
    if len(all_arr) >= 3:
        valid = all_arr[np.isfinite(all_arr)]
        if len(valid) >= 3:
            try:
                sw_stat, sw_p = shapiro(valid)
            except:
                pass
        if len(valid) >= 20:
            try:
                ad_result = anderson(valid, dist='norm')
                if ad_result:
                    ad_pass = ad_result.statistic < ad_result.critical_values[2]
            except:
                pass
        if len(valid) > 0:
            sorted_data = np.sort(valid)
            n_vals = len(sorted_data)
            theoretical_quantiles = norm.ppf((np.arange(1, n_vals+1) - 0.5) / n_vals)
            qq_points = [[round(theoretical_quantiles[i], 4), round(sorted_data[i], 4)] for i in range(n_vals)]

    title_map = {
        'I-MR': ('单值控制图', '移动极差图'),
        'Xbar-R': ('均值-极差控制图', 'R控制图'),
        'Xbar-S': ('均值-标准差控制图', 'S控制图')
    }
    title_xbar, title_r = title_map.get(chart_type, ('控制图', '极差/标准差图'))

    return {
        "chart_type": chart_type,
        "message": message,
        "title_xbar": title_xbar,
        "title_r": title_r,
        "labels": batch_labels,
        "dates": batch_dates,
        "xbar": [round(v, 4) for v in xbar],
        "r": [round(v, 4) if v is not None else None for v in plot_r],
        "r_label": r_label,
        "ucl_x": round(ucl_x, 4),
        "lcl_x": round(lcl_x, 4),
        "cl_x": round(center_x, 4),
        "ucl_r": round(ucl_r, 4),
        "lcl_r": round(lcl_r, 4) if lcl_r is not None else None,
        "cl_r": round(center_r, 4),
        "alarm_x": list(set(alarm_x)),
        "alarm_r": list(set(alarm_r)),
        "rules_violations": rules_violations,
        "all_values": [round(v, 4) for v in all_values],
        "mean": round(mean_total, 4),
        "std_within": round(sigma_within, 6) if sigma_within else 0,
        "std_overall": round(std_total, 6) if std_total else 0,
        "cp": round(cp, 4) if cp is not None else None,
        "cpk": round(cpk, 4) if cpk is not None else None,
        "pp": round(pp, 4) if pp is not None else None,
        "ppk": round(ppk, 4) if ppk is not None else None,
        "cpm": round(cpm_val, 4) if cpm_val is not None else None,
        "cpmk": round(cpmk_val, 4) if cpmk_val is not None else None,
        "z_bench": round(z_bench, 4) if z_bench is not None else None,
        "ppm_total": round(ppm_total, 0) if ppm_total is not None else None,
        "ppm_usl": round(ppm_usl, 0) if ppm_usl is not None else None,
        "ppm_lsl": round(ppm_lsl, 0) if ppm_lsl is not None else None,
        "usl": ctq_usl,
        "lsl": ctq_lsl,
        "target": ctq_target,
        "sw_stat": round(sw_stat, 4) if sw_stat else None,
        "sw_p": round(sw_p, 4) if sw_p else None,
        "ad_stat": round(ad_result.statistic, 4) if ad_result else None,
        "ad_crit_5": round(ad_result.critical_values[2], 4) if ad_result else None,
        "ad_pass": bool(ad_pass) if ad_pass is not None else None,
        "qq_points": qq_points,
        "ctq_name": f"{production_data[0].product_item or ''} - {production_data[0].ctq_name}" if production_data else "",
        "time_range": f"{batch_dates[0]} ~ {batch_dates[-1]}" if batch_dates else ""
    }