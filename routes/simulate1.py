# routes/simulate.py
import json
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, send_file
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
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

    # 动态获取品项列表（从数据库已有生产数据或 CTQ 配置）
    existing_items = db.session.query(ProductionData.product_item).distinct()\
        .filter(ProductionData.product_item.isnot(None)).all()
    product_items = sorted(list(set([i[0] for i in existing_items if i[0]])))
    # 如果没有数据，使用 CTQ 配置中的品项（非空）
    if not product_items:
        ctq_items = db.session.query(CTQConfig.product_item).distinct().filter(CTQConfig.product_item.isnot(None)).all()
        product_items = sorted([i[0] for i in ctq_items if i[0]])

    # 从 JSON 加载因子库（预置）
    factor_lib_path = Path(__file__).parent.parent / 'data' / 'factor_library.json'
    if factor_lib_path.exists():
        with open(factor_lib_path, 'r', encoding='utf-8') as f:
            factor_library = json.load(f)
    else:
        factor_library = {}
        logger.warning("factor_library.json 不存在，模拟生成器将无法使用预置因子库")

    return render_template('simulate.html',
                           ctqs_json=ctqs_json,
                           product_items=product_items,
                           factor_library=json.dumps(factor_library, ensure_ascii=False))


@simulate_bp.route('/preview', methods=['POST'])
def preview():
    config = request.get_json()
    if not config:
        return jsonify({'error': '无效配置'}), 400

    preview_cfg = config.copy()
    preview_cfg['batch_count'] = min(20, config.get('batch_count', 50))
    preview_cfg['samples_per_batch'] = min(config.get('samples_per_batch', 5), 5)
    preview_cfg['preview_mode'] = True

    # 验证公式非空
    for ctq_id in config.get('ctq_ids', []):
        has_features = config.get('ctq_features', {}).get(str(ctq_id), [])
        has_formula = config.get('ctq_formulas', {}).get(str(ctq_id), '').strip()
        if has_features and not has_formula:
            ctq = CTQConfig.query.get(ctq_id)
            name = ctq.ctq_name if ctq else ctq_id
            return jsonify({'error': f'CTQ「{name}」已配置特征但未填写公式，请补全公式后再预览'}), 400

    try:
        prod_df, feat_df = generate_mock_data(preview_cfg)
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
            ppk = None
            if usl and lsl and std_val > 0:
                cpu = (usl - mean_val) / (3 * std_val)
                cpl = (mean_val - lsl) / (3 * std_val)
                ppk = round(min(cpu, cpl), 4)

            # 计算真实R²：通过原始公式值与实测值的相关系数平方
            r2_est = 0.0
            formula_str = config.get('ctq_formulas', {}).get(str(ctq_id), '').strip()
            if formula_str and not feat_df.empty:
                ctq_feat = feat_df[feat_df['ctq_id'] == ctq_id]
                if not ctq_feat.empty:
                    pivot = ctq_feat.pivot(index='batch_no', columns='feature_name', values='feature_value')
                    numeric_cols = pivot.select_dtypes(include=[np.number]).columns
                    pivot = pivot[numeric_cols].astype(float)
                    y_series = df_ctq.set_index('batch_no')['measured_value']
                    common_idx = pivot.index.intersection(y_series.index)
                    if len(common_idx) >= 5:
                        X = pivot.loc[common_idx]
                        y = y_series.loc[common_idx]
                        raw_vals = []
                        for idx, row in X.iterrows():
                            feat_dict = row.to_dict()
                            clean_dict = {k.strip(): v for k, v in feat_dict.items()}
                            try:
                                rv = SafeEvaluator.evaluate(formula_str, clean_dict)
                                raw_vals.append(rv)
                            except:
                                raw_vals.append(np.nan)
                        raw_series = pd.Series(raw_vals, index=X.index).dropna()
                        if len(raw_series) >= 5:
                            corr = np.corrcoef(raw_series, y.loc[raw_series.index])[0, 1]
                            r2_est = round(corr ** 2, 4)

            ctq_stats[ctq.ctq_name] = {
                'ppk': ppk,
                'r2_est': r2_est,
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
        clean_vars = {k.strip(): v for k, v in sample_values.items()}
        result = SafeEvaluator.evaluate(formula, clean_vars)
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

    for ctq_id in config.get('ctq_ids', []):
        has_features = config.get('ctq_features', {}).get(str(ctq_id), [])
        has_formula = config.get('ctq_formulas', {}).get(str(ctq_id), '').strip()
        if has_features and not has_formula:
            ctq = CTQConfig.query.get(ctq_id)
            name = ctq.ctq_name if ctq else ctq_id
            return jsonify({'error': f'CTQ「{name}」已配置特征但未填写公式，请补全公式后再生成'}), 400

    try:
        prod_df, feat_df = generate_mock_data(config)
        if config.get('auto_import'):
            import_mode = config.get('import_mode', 'replace')
            import_to_db(prod_df, feat_df, import_mode)
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
    force_ppk_config = config.get('force_cpk_config', {})
    ctq_features_map = config.get('ctq_features', {})

    batches = []
    for i in range(batch_count):
        produce_date = start_date + timedelta(days=i // 3)
        batch_no = f"MOCK{produce_date.strftime('%Y%m%d')}{i+1:03d}"
        product_item = product_items[i % len(product_items)] if product_items else "原味"
        product_line = product_lines[i % len(product_lines)] if product_lines else "1号线"
        work_shift = shifts[i % len(shifts)] if shifts else "早班"
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
    raw_values_per_sample = []

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
                        feature_rows.append({
                            'batch_no': batch['batch_no'],
                            'ctq_id': ctq.ctq_id,
                            'feature_name': name,
                            'feature_value': round(val, 4),
                            'raw_value': None,
                            'feature_type': 'numeric'
                        })
                    else:
                        categories = feat.get('categories', [])
                        weights = feat.get('weights', [1]*len(categories))
                        if sum(weights) == 0:
                            weights = [1]*len(categories)
                        probs = np.array(weights) / sum(weights)
                        cat = np.random.choice(categories, p=probs)
                        feat_dict[name] = cat
                        feature_rows.append({
                            'batch_no': batch['batch_no'],
                            'ctq_id': ctq.ctq_id,
                            'feature_name': name,
                            'feature_value': None,
                            'raw_value': cat,
                            'feature_type': 'categorical'
                        })
                raw_val = None
                if formula:
                    safe_vars = {}
                    for k, v in feat_dict.items():
                        val = v.item() if hasattr(v, 'item') else v
                        safe_vars[k] = val
                        stripped_key = k.strip()
                        if stripped_key != k:
                            safe_vars[stripped_key] = val
                    try:
                        raw_val = SafeEvaluator.evaluate(formula, safe_vars)
                    except Exception as e:
                        logger.warning(f"Formula error: {e}")
                raw_values_per_sample.append((ctq_id, batch['batch_no'], sample_idx, raw_val))

    # PPK 强制：线性缩放 + 适量噪声
    ppk_params = {}
    for ctq in ctq_list:
        ctq_id = str(ctq.ctq_id)
        cfg = force_ppk_config.get(ctq_id, {})
        enabled = cfg.get('enabled', False)
        if enabled:
            target_ppk = cfg.get('target_cpk', 1.33)
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
                        raw_std = 1e-6
                    if raw_std > target_std:
                        scale = target_std / raw_std
                        signal_scaled_std = target_std
                    else:
                        scale = 1.0
                        signal_scaled_std = raw_std
                    noise_var = max(0, target_std**2 - signal_scaled_std**2)
                    noise_std = np.sqrt(noise_var)
                    shift = target_mean - scale * raw_mean
                    ppk_params[ctq_id] = {
                        'enabled': True,
                        'scale': scale,
                        'shift': shift,
                        'noise_std': noise_std
                    }
        if not enabled:
            ppk_params[ctq_id] = {'enabled': False}

    production_rows = []
    raw_map = {}
    for (cid, batch_no, sample_idx, rv) in raw_values_per_sample:
        raw_map[(cid, batch_no, sample_idx)] = rv

    default_target_cpk = config.get('target_cpk', 1.0)
    default_bias_tendency = config.get('bias_tendency', 0.0)

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
                    scaled_val = ppk_info['scale'] * raw_val
                    value = scaled_val + ppk_info['shift'] + np.random.normal(0, ppk_info['noise_std'])
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
    feat_df = pd.DataFrame(feature_rows)
    return prod_df, feat_df


def import_to_db(prod_df, feat_df, import_mode='replace'):
    if import_mode == 'replace_all':
        ProductionData.query.filter(ProductionData.batch_no.like('MOCK%')).delete()
        CtqFeatureValue.query.filter(CtqFeatureValue.batch_no.like('MOCK%')).delete()
    elif import_mode == 'replace':
        ctq_ids = prod_df['ctq_id'].unique().tolist()
        ProductionData.query.filter(
            ProductionData.batch_no.like('MOCK%'),
            ProductionData.ctq_id.in_(ctq_ids)
        ).delete()
        CtqFeatureValue.query.filter(
            CtqFeatureValue.batch_no.like('MOCK%'),
            CtqFeatureValue.ctq_id.in_(ctq_ids)
        ).delete()
    db.session.commit()

    prod_objects = []
    for _, row in prod_df.iterrows():
        row_dict = row.to_dict()
        if isinstance(row_dict.get('produce_date'), pd.Timestamp):
            row_dict['produce_date'] = row_dict['produce_date'].date()
        prod = ProductionData(**row_dict)
        prod_objects.append(prod)
    if prod_objects:
        db.session.bulk_save_objects(prod_objects)

    feat_objects = []
    for _, row in feat_df.iterrows():
        feat_dict = {
            'batch_no': row['batch_no'],
            'ctq_id': row['ctq_id'],
            'feature_name': row['feature_name'],
            'feature_value': row['feature_value'] if pd.notna(row['feature_value']) else None,
            'raw_value': row['raw_value'] if pd.notna(row['raw_value']) else None,
            'feature_type': row['feature_type']
        }
        feat = CtqFeatureValue(**feat_dict)
        feat_objects.append(feat)
    if feat_objects:
        db.session.bulk_save_objects(feat_objects)

    db.session.commit()
    clear_all_caches()