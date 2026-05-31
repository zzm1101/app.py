# routes/simulate.py
import json
import math
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, send_file
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from models.database import db
from models.models import CTQConfig, ProductionData
from models.influence_models import CtqFeatureValue, SystemSetting
from extensions import clear_all_caches
from utils.safe_eval import SafeEvaluator
import logging

logger = logging.getLogger(__name__)
simulate_bp = Blueprint('simulate', __name__, url_prefix='/simulate')


# ==================== 页面路由 ====================
@simulate_bp.route('/')
def index():
    ctqs = CTQConfig.query.filter(CTQConfig.status == "启用").all()
    ctq_list = [{
        'ctq_id': c.ctq_id,
        'ctq_name': c.ctq_name,
        'product_item': c.product_item or ''
    } for c in ctqs]
    ctqs_json = json.dumps(ctq_list, ensure_ascii=False)

    # 动态获取品项列表
    existing_items = db.session.query(ProductionData.product_item).distinct()\
        .filter(ProductionData.product_item.isnot(None)).all()
    product_items = sorted(list(set([i[0] for i in existing_items if i[0]])))
    if not product_items:
        ctq_items = db.session.query(CTQConfig.product_item).distinct().filter(CTQConfig.product_item.isnot(None)).all()
        product_items = sorted([i[0] for i in ctq_items if i[0]])

    # 加载预置因子库
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


# ==================== 预览与公式测试 ====================
@simulate_bp.route('/preview', methods=['POST'])
def preview():
    config = request.get_json()
    if not config:
        return jsonify({'error': '无效配置'}), 400

    preview_cfg = config.copy()
    preview_cfg['batch_count'] = min(20, config.get('batch_count', 50))
    preview_cfg['samples_per_batch'] = min(config.get('samples_per_batch', 5), 5)
    preview_cfg['preview_mode'] = True

    # 检查公式完整性
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
                            if np.isnan(r2_est) or np.isinf(r2_est):
                                r2_est = None

            ctq_stats[ctq.ctq_name] = {
                'ppk': ppk,
                'r2_est': r2_est,
                'mean': round(mean_val, 4),
                'std': round(std_val, 4)
            }

        # 清理所有统计值中的 NaN/Inf，转为 null (JSON 合法)
        for name, stats in ctq_stats.items():
            for k, v in stats.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    stats[k] = None

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


# ==================== 配置持久化 API ====================
@simulate_bp.route('/api/save_config', methods=['POST'])
def save_simulate_config():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '无效的配置数据'}), 400

        setting = SystemSetting.query.filter_by(key='simulate_config').first()
        if setting:
            setting.value = json.dumps(data, ensure_ascii=False)
        else:
            setting = SystemSetting(key='simulate_config', value=json.dumps(data, ensure_ascii=False))
            db.session.add(setting)
        db.session.commit()
        return jsonify({'status': 'success', 'message': '配置已保存到服务器'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@simulate_bp.route('/api/load_config', methods=['GET'])
def load_simulate_config():
    try:
        setting = SystemSetting.query.filter_by(key='simulate_config').first()
        if setting:
            config = json.loads(setting.value)
            return jsonify({'status': 'success', 'config': config})
        else:
            return jsonify({'status': 'success', 'config': None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@simulate_bp.route('/api/save_libraries', methods=['POST'])
def save_custom_libraries():
    try:
        data = request.get_json()
        libraries = data.get('libraries', [])
        setting = SystemSetting.query.filter_by(key='custom_factor_libraries').first()
        if setting:
            setting.value = json.dumps(libraries, ensure_ascii=False)
        else:
            setting = SystemSetting(key='custom_factor_libraries', value=json.dumps(libraries, ensure_ascii=False))
            db.session.add(setting)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f'已保存 {len(libraries)} 个自定义因子库'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@simulate_bp.route('/api/load_libraries', methods=['GET'])
def load_custom_libraries():
    try:
        setting = SystemSetting.query.filter_by(key='custom_factor_libraries').first()
        if setting:
            libraries = json.loads(setting.value)
            return jsonify({'status': 'success', 'libraries': libraries})
        else:
            return jsonify({'status': 'success', 'libraries': []})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==================== 模拟数据生成核心函数（带详细调试） ====================
def generate_mock_data(config, preview_mode=False):
    """
    生成模拟生产数据和特征数据
    """
    # ---------- 1. 解析配置 ----------
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

    # 统一将键转为字符串，避免类型不匹配
    ctq_formulas_raw = config.get('ctq_formulas', {})
    ctq_formulas = {str(k): v for k, v in ctq_formulas_raw.items()}
    ctq_features_raw = config.get('ctq_features', {})
    ctq_features_map = {str(k): v for k, v in ctq_features_raw.items()}
    force_ppk_config_raw = config.get('force_cpk_config', {})
    force_ppk_config = {str(k): v for k, v in force_ppk_config_raw.items()}

    # 调试输出
    print("\n" + "="*70)
    print("[DEBUG] generate_mock_data 接收到的配置:")
    print(f"  ctq_ids: {ctq_ids}")
    print(f"  ctq_formulas 键: {list(ctq_formulas.keys())}")
    for cid, f in ctq_formulas.items():
        print(f"    CTQ {cid} 公式: {f[:80]}...")
    print(f"  ctq_features_map 键: {list(ctq_features_map.keys())}")
    print("="*70)

    # ---------- 2. 生成批次基础信息 ----------
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
    raw_values_per_sample = []  # (ctq_id, batch_no, sample_idx, raw_value)

    # ---------- 3. 生成特征值并计算公式原始值 ----------
    for batch in batches:
        for ctq in ctq_list:
            ctq_id = str(ctq.ctq_id)
            feat_list = ctq_features_map.get(ctq_id, [])
            formula = ctq_formulas.get(ctq_id)

            for sample_idx in range(samples_per_batch):
                feat_dict = {}
                # 生成特征
                for feat in feat_list:
                    name = feat['name']
                    ftype = feat['type']
                    if ftype == 'numeric':
                        dist = feat.get('distribution', 'normal')
                        try:
                            if dist == 'normal':
                                mu = float(feat.get('mean', 0))
                                sigma = float(feat.get('std', 1))
                                if sigma <= 0:
                                    sigma = 0.001
                                val = np.random.normal(mu, sigma)
                            elif dist == 'uniform':
                                low = float(feat.get('min', 0))
                                high = float(feat.get('max', 1))
                                if high <= low:
                                    high = low + 0.001
                                val = np.random.uniform(low, high)
                            elif dist == 'lognormal':
                                mean_orig = float(feat.get('mean', 1))
                                std_orig = float(feat.get('std', 0.5))
                                if mean_orig <= 0:
                                    mean_orig = 1.0
                                if std_orig <= 0:
                                    std_orig = 0.001
                                mu_log = np.log(mean_orig**2 / np.sqrt(mean_orig**2 + std_orig**2))
                                sigma_log = np.sqrt(np.log(1 + (std_orig**2 / mean_orig**2)))
                                val = np.random.lognormal(mu_log, sigma_log)
                            else:
                                val = np.random.normal(0, 1)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"特征 {name} 参数解析失败，使用默认正态分布。错误: {e}")
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
                    else:  # categorical
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

                # 计算公式原始值
                raw_val = None
                if formula:
                    safe_vars = {}
                    for k, v in feat_dict.items():
                        val = v.item() if hasattr(v, 'item') else v
                        safe_vars[k] = val
                        stripped_key = k.strip()
                        if stripped_key != k:
                            safe_vars[stripped_key] = val
                    # 第一个样本打印调试信息
                    if batch == batches[0] and sample_idx == 0:
                        print(f"\n[DEBUG] CTQ {ctq_id} 公式: {formula}")
                        print(f"[DEBUG] 可用变量: {list(safe_vars.keys())}")
                    try:
                        raw_val = SafeEvaluator.evaluate(formula, safe_vars)
                        if batch == batches[0] and sample_idx == 0:
                            print(f"[DEBUG] 公式计算结果: {raw_val}")
                    except Exception as e:
                        if batch == batches[0] and sample_idx == 0:
                            print(f"[ERROR] 公式计算失败: {e}")
                        raw_val = None
                else:
                    if batch == batches[0] and sample_idx == 0:
                        print(f"[INFO] CTQ {ctq_id} 没有公式，将使用 PPK 模型生成数据")

                raw_values_per_sample.append((ctq_id, batch['batch_no'], sample_idx, raw_val))

    # ---------- 4. 处理强制 PPK（简化，默认不启用）----------
    ppk_params = {}
    for ctq in ctq_list:
        ctq_id = str(ctq.ctq_id)
        cfg = force_ppk_config.get(ctq_id, {})
        enabled = cfg.get('enabled', False)
        ppk_params[ctq_id] = {'enabled': False}

    # ---------- 5. 生成最终生产数据 ----------
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
            formula = ctq_formulas.get(ctq_id)
            ppk_info = ppk_params.get(ctq_id, {'enabled': False})

            for sample_idx in range(samples_per_batch):
                key = (ctq_id, batch['batch_no'], sample_idx)
                raw_val = raw_map.get(key)

                # 决定最终测量值
                if formula and raw_val is not None:
                    value = raw_val
                    if batch == batches[0] and sample_idx == 0:
                        print(f"[DEBUG] CTQ {ctq_id} 使用公式值: {value}")
                else:
                    if batch == batches[0] and sample_idx == 0:
                        reason = "无公式" if not formula else "公式计算失败"
                        print(f"[WARN] CTQ {ctq_id} {reason}，使用 PPK 模型。目标值={target}")
                    # 根据 PPK 和倾向生成
                    offset = (usl - lsl) * default_bias_tendency / 2
                    mean = target + offset
                    min_dist = min(usl - mean, mean - lsl)
                    if min_dist <= 0 or default_target_cpk <= 0:
                        sigma = (usl - lsl) / 6
                    else:
                        sigma = min_dist / (3 * default_target_cpk)
                    sigma = max(sigma, 0.001)
                    value = np.random.normal(mean, sigma)

                # 异常值处理
                if outlier_ratio > 0 and np.random.rand() < outlier_ratio:
                    if np.random.rand() < 0.5:
                        value = usl * (1 + np.random.uniform(0.1, 0.5))
                    else:
                        value = lsl * (1 - np.random.uniform(0.1, 0.5))

                # 只保底非负，不再强制修正到规格限附近，避免掩盖公式小数值的真实波动
                value = np.clip(value, 0, None)

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


# ==================== 生成数据主路由 ====================
@simulate_bp.route('/generate', methods=['POST'])
def generate():
    config = request.get_json()
    if not config:
        return jsonify({'error': '无效的配置'}), 400

    # 检查公式完整性
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
            # 导出 Excel
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


def import_to_db(prod_df, feat_df, import_mode='replace'):
    """将模拟数据导入数据库"""
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