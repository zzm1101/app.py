# routes/simulate.py
import json
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, send_file
from io import BytesIO
from datetime import datetime, timedelta
from models.database import db
from models.models import CTQConfig, ProductionData
from models.influence_models import CtqFeatureValue
from extensions import clear_all_caches
from utils.safe_eval import SafeEvaluator
import logging

logger = logging.getLogger(__name__)
simulate_bp = Blueprint('simulate', __name__, url_prefix='/simulate')


@simulate_bp.route('/')
def index():
    ctqs = CTQConfig.query.filter(CTQConfig.status == "启用").all()
    ctq_list = [{
        'ctq_id': c.ctq_id,
        'ctq_name': c.ctq_name,
        'product_item': c.product_item or ''
    } for c in ctqs]
    ctqs_json = json.dumps(ctq_list, ensure_ascii=False)

    existing_items = db.session.query(ProductionData.product_item).distinct()\
        .filter(ProductionData.product_item.isnot(None)).all()
    product_items = sorted(list(set([i[0] for i in existing_items if i[0]]))) or ["原味", "草莓", "蓝莓", "黄桃", "高蛋白"]

    # 预置影响因子库（注意：使用 log10 而非 np.log10）
    factor_library = {
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

    return render_template('simulate.html',
                           ctqs_json=ctqs_json,
                           product_items=product_items,
                           factor_library=json.dumps(factor_library, ensure_ascii=False))


@simulate_bp.route('/preview', methods=['POST'])
def preview():
    """快速预览：基于当前配置生成少量数据，返回PPK和预估R²"""
    config = request.get_json()
    if not config:
        return jsonify({'error': '无效配置'}), 400

    preview_cfg = config.copy()
    preview_cfg['batch_count'] = min(20, config.get('batch_count', 50))
    preview_cfg['samples_per_batch'] = min(config.get('samples_per_batch', 5), 5)   # 限制预览样品数
    preview_cfg['preview_mode'] = True

    try:
        prod_df, _ = generate_mock_data(preview_cfg)
        if prod_df.empty:
            return jsonify({'error': '预览数据生成失败'}), 500

        ctq_stats = {}
        for ctq_id in config.get('ctq_ids', []):
            ctq = CTQConfig.query.get(ctq_id)
            if not ctq:
                continue
            df_ctq = prod_df[prod_df['ctq_id'] == ctq_id]
            if df_ctq.empty:
                continue
            values = df_ctq['measured_value'].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            usl, lsl = ctq.usl, ctq.lsl
            if usl and lsl and std_val > 0:
                cpu = (usl - mean_val) / (3 * std_val)
                cpl = (mean_val - lsl) / (3 * std_val)
                ppk = round(min(cpu, cpl), 4)
            else:
                ppk = None

            noise_std = config.get('force_cpk_config', {}).get(str(ctq_id), {}).get('noise_std', 0)
            if noise_std == 0:
                r2_est = 0.99
            else:
                r2_est = max(0, 1 - (noise_std ** 2) / (std_val ** 2)) if std_val > 0 else 0.5
            ctq_stats[ctq.ctq_name] = {
                'ppk': ppk,
                'r2_est': round(r2_est, 4),
                'mean': round(mean_val, 4),
                'std': round(std_val, 4)
            }
        return jsonify({'status': 'success', 'stats': ctq_stats})
    except Exception as e:
        logger.exception("Preview error")
        return jsonify({'error': str(e)}), 500


@simulate_bp.route('/test_formula', methods=['POST'])
def test_formula():
    data = request.get_json()
    formula = data.get('formula', '').strip()
    sample_values = data.get('sample_values', {})
    if not formula:
        return jsonify({'status': 'error', 'message': '公式为空'})
    try:
        result = SafeEvaluator.evaluate(formula, sample_values)
        if isinstance(result, (np.floating, np.integer)):
            result = float(result)
        elif isinstance(result, np.ndarray):
            result = result.tolist()
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@simulate_bp.route('/generate', methods=['POST'])
def generate():
    config = request.get_json()
    if not config:
        return jsonify({'error': '无效的配置'}), 400
    try:
        prod_df, feat_df = generate_mock_data(config)
        if config.get('auto_import'):
            import_to_db(prod_df, feat_df, config.get('clear_old_mock', True))
            return jsonify({
                'status': 'success',
                'message': f'已导入 {len(prod_df)} 条生产数据，{len(feat_df)} 条特征数据',
                'prod_count': len(prod_df),
                'feat_count': len(feat_df)
            })
        else:
            col_map = {
                'batch_no': '生产批次号',
                'produce_date': '生产日期',
                'product_line': '生产线',
                'work_shift': '生产班次',
                'product_item': '品项',
                'sample_no': '样品编号',
                'ctq_id': 'CTQ编号',
                'ctq_name': 'CTQ名称',
                'measured_value': '实测值',
                'batch_quantity': '批次生产数量',
                'storage_days': '存储天数',
                'storage_temp': '存储温度(℃)',
                'inspector': '检验员',
                'production_week': '生产周',
                'production_month': '生产月',
                'production_year_month': '生产年月',
            }
            prod_df_export = prod_df.rename(columns=col_map)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                prod_df_export.to_excel(writer, sheet_name='生产数据', index=False)
                if not feat_df.empty:
                    feat_df.to_excel(writer, sheet_name='特征数据', index=False)
            output.seek(0)
            return send_file(
                output,
                download_name=f'mock_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                as_attachment=True,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    except Exception as e:
        logger.exception("Generate error")
        return jsonify({'error': str(e)}), 500


def generate_mock_data(config, preview_mode=False):
    """核心生成函数"""
    ctq_ids = config['ctq_ids']
    batch_count = config['batch_count']
    samples_per_batch = config['samples_per_batch']
    start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
    product_items = config['product_items']
    product_lines = config['product_lines']
    shifts = config['shifts']
    batch_qty_min = config['batch_qty_min']
    batch_qty_max = config['batch_qty_max']
    outlier_ratio = config.get('outlier_ratio', 0.0)
    random_seed = config.get('random_seed')
    if random_seed:
        np.random.seed(random_seed)

    ctq_list = CTQConfig.query.filter(CTQConfig.ctq_id.in_(ctq_ids)).all()
    ctq_map = {c.ctq_id: c for c in ctq_list}

    ctq_formulas = config.get('ctq_formulas', {})
    force_ppk_config = config.get('force_cpk_config', {})   # 字段名保持向后兼容，内部理解为 PPK
    ctq_features_map = config.get('ctq_features', {})

    # 生成批次元数据
    batches = []
    for i in range(batch_count):
        produce_date = start_date + timedelta(days=i // 3)
        batch_no = f"MOCK{produce_date.strftime('%Y%m%d')}{i+1:03d}"
        product_item = product_items[i % len(product_items)]
        product_line = product_lines[i % len(product_lines)]
        work_shift = shifts[i % len(shifts)]
        batch_quantity = np.random.randint(batch_qty_min, batch_qty_max + 1)
        batches.append({
            'batch_no': batch_no,
            'produce_date': produce_date,
            'product_item': product_item,
            'product_line': product_line,
            'work_shift': work_shift,
            'batch_quantity': batch_quantity,
            'year_month': produce_date.strftime('%Y-%m'),
            'week': produce_date.isocalendar()[1],
            'month': produce_date.month
        })

    feature_rows = []
    raw_values_per_sample = []   # 每个元素为 (ctq_id, batch_no, sample_idx, raw_val)

    # 第一步：生成所有样品级的特征值，并计算公式原始值
    for batch in batches:
        for ctq in ctq_list:
            ctq_id = str(ctq.ctq_id)
            feat_list = ctq_features_map.get(ctq_id, [])
            formula = ctq_formulas.get(ctq_id)
            for sample_idx in range(samples_per_batch):
                feat_dict = {}
                for feat in feat_list:
                    name = feat['name']
                    ftype = feat['type']
                    if ftype == 'numeric':
                        dist = feat.get('distribution', 'normal')
                        if dist == 'normal':
                            mu = feat.get('mean', 0)
                            sigma = feat.get('std', 1)
                            val = np.random.normal(mu, sigma)
                        elif dist == 'uniform':
                            low = feat.get('min', 0)
                            high = feat.get('max', 1)
                            val = np.random.uniform(low, high)
                        elif dist == 'lognormal':
                            mean_orig = feat.get('mean', 1)
                            std_orig = feat.get('std', 0.5)
                            mu_log = np.log(mean_orig ** 2 / np.sqrt(mean_orig ** 2 + std_orig ** 2))
                            sigma_log = np.sqrt(np.log(1 + (std_orig ** 2 / mean_orig ** 2)))
                            val = np.random.lognormal(mu_log, sigma_log)
                        else:
                            val = np.random.normal(0, 1)
                        feat_dict[name] = val
                        if not preview_mode:
                            feature_rows.append({
                                'batch_no': batch['batch_no'],
                                'ctq_id': ctq.ctq_id,
                                'ctq_name': ctq.ctq_name,
                                'feature_name': name,
                                'feature_value': round(val, 4),
                                'raw_value': None,
                                'feature_type': 'numeric'
                            })
                    else:  # categorical
                        categories = feat.get('categories', [])
                        weights = feat.get('weights', [1]*len(categories))
                        if sum(weights) == 0:
                            weights = [1]*len(categories)
                        probs = np.array(weights) / sum(weights)
                        cat = np.random.choice(categories, p=probs)
                        feat_dict[name] = cat
                        if not preview_mode:
                            feature_rows.append({
                                'batch_no': batch['batch_no'],
                                'ctq_id': ctq.ctq_id,
                                'ctq_name': ctq.ctq_name,
                                'feature_name': name,
                                'feature_value': None,
                                'raw_value': cat,
                                'feature_type': 'categorical'
                            })
                # 计算公式原始值
                raw_val = None
                if formula:
                    safe_vars = {k: (v.item() if hasattr(v, 'item') else v) for k, v in feat_dict.items()}
                    try:
                        raw_val = SafeEvaluator.evaluate(formula, safe_vars)
                    except Exception as e:
                        logger.warning(f"Formula error: {e}")
                raw_values_per_sample.append((ctq_id, batch['batch_no'], sample_idx, raw_val))

    # 第二步：对每个 CTQ，如果需要 PPK 强制，则收集所有原始值并计算缩放参数
    ppk_params = {}
    for ctq in ctq_list:
        ctq_id = str(ctq.ctq_id)
        cfg = force_ppk_config.get(ctq_id, {})
        enabled = cfg.get('enabled', False)
        if enabled:
            target_ppk = cfg.get('target_cpk', 1.33)
            noise_std = cfg.get('noise_std', 0.0)
            target_mean = ctq.target_m
            usl, lsl = ctq.usl, ctq.lsl
            if usl is None or lsl is None:
                enabled = False
            else:
                delta = min(usl - target_mean, target_mean - lsl)
                target_std = delta / (3 * target_ppk)
                raw_vals = [rv for (cid, _, _, rv) in raw_values_per_sample if cid == ctq_id and rv is not None]
                if len(raw_vals) < 2:
                    enabled = False
                    logger.warning(f"CTQ {ctq.ctq_name} 原始有效值不足，禁用 PPK 强制")
                else:
                    raw_mean = np.mean(raw_vals)
                    raw_std = np.std(raw_vals, ddof=1)
                    if raw_std == 0:
                        raw_std = 1
                    scale = target_std / raw_std
                    ppk_params[ctq_id] = {
                        'enabled': True,
                        'target_mean': target_mean,
                        'raw_mean': raw_mean,
                        'scale': scale,
                        'noise_std': noise_std
                    }
        if not enabled:
            ppk_params[ctq_id] = {'enabled': False}

    # 第三步：生成生产数据（实测值）
    production_rows = []
    default_target_cpk = config.get('target_cpk', 1.0)
    default_bias_tendency = config.get('bias_tendency', 0.0)

    raw_map = {}
    for (cid, batch_no, sample_idx, rv) in raw_values_per_sample:
        raw_map[(cid, batch_no, sample_idx)] = rv

    for batch in batches:
        for ctq in ctq_list:
            ctq_id = str(ctq.ctq_id)
            usl, lsl = ctq.usl, ctq.lsl
            target = ctq.target_m
            if usl is None or lsl is None:
                continue
            ppk_info = ppk_params.get(ctq_id, {'enabled': False})
            for sample_idx in range(samples_per_batch):
                key = (ctq_id, batch['batch_no'], sample_idx)
                raw_val = raw_map.get(key)
                if raw_val is not None and ppk_info['enabled']:
                    scaled = ppk_info['target_mean'] + ppk_info['scale'] * (raw_val - ppk_info['raw_mean'])
                    value = scaled + np.random.normal(0, ppk_info['noise_std'])
                elif raw_val is not None:
                    value = raw_val
                else:
                    offset = (usl - lsl) * default_bias_tendency / 2
                    mean = target + offset
                    min_dist = min(usl - mean, mean - lsl)
                    if min_dist <= 0 or default_target_cpk <= 0:
                        sigma = (usl - lsl) / 6
                    else:
                        sigma = min_dist / (3 * default_target_cpk)
                    sigma = max(sigma, 0.001)
                    value = np.random.normal(mean, sigma)

                if outlier_ratio > 0 and np.random.rand() < outlier_ratio:
                    if np.random.rand() < 0.5:
                        value = usl * (1 + np.random.uniform(0.1, 0.5))
                    else:
                        value = lsl * (1 - np.random.uniform(0.1, 0.5))
                value = np.clip(value, lsl * 0.5, usl * 1.5)

                production_rows.append({
                    'batch_no': batch['batch_no'],
                    'produce_date': batch['produce_date'].date(),
                    'product_line': batch['product_line'],
                    'work_shift': batch['work_shift'],
                    'product_item': batch['product_item'],
                    'sample_no': f"S{batch['batch_no']}-{sample_idx+1}",
                    'ctq_id': ctq.ctq_id,
                    'ctq_name': ctq.ctq_name,
                    'measured_value': round(value, 4),
                    'batch_quantity': batch['batch_quantity'],
                    'storage_days': 0,
                    'storage_temp': 4.0,
                    'inspector': '模拟生成',
                    'production_week': batch['week'],
                    'production_month': batch['month'],
                    'production_year_month': batch['year_month']
                })

    prod_df = pd.DataFrame(production_rows)
    feat_df = pd.DataFrame(feature_rows) if not preview_mode else pd.DataFrame()
    return prod_df, feat_df


def import_to_db(prod_df, feat_df, clear_old_mock=True):
    if clear_old_mock:
        ProductionData.query.filter(ProductionData.batch_no.like('MOCK%')).delete()
        CtqFeatureValue.query.filter(CtqFeatureValue.batch_no.like('MOCK%')).delete()
        db.session.commit()

    prod_objects = []
    for _, row in prod_df.iterrows():
        if isinstance(row['produce_date'], datetime):
            row['produce_date'] = row['produce_date'].date()
        prod = ProductionData(**row.to_dict())
        prod_objects.append(prod)
    if prod_objects:
        db.session.bulk_save_objects(prod_objects)

    feat_objects = []
    for _, row in feat_df.iterrows():
        feat = CtqFeatureValue(**row.to_dict())
        feat_objects.append(feat)
    if feat_objects:
        db.session.bulk_save_objects(feat_objects)

    db.session.commit()
    clear_all_caches()