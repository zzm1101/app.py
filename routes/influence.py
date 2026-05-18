# routes/influence.py
import json
import os
import uuid
import time
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from sklearn.model_selection import cross_validate, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from models.database import db
from models.models import CTQConfig, ProductionData
from models.influence_models import CtqFeatureValue, CtqTargetOverride, MlModelMetadata, MlModelType
from utils import normalize_product_item

influence_bp = Blueprint('influence', __name__, url_prefix='/influence')

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

_executor = ThreadPoolExecutor(max_workers=2)
_tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_class(model_type):
    if model_type == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor
    elif model_type == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor
    elif model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        return LinearRegression
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_feature_importance(model, model_type, feature_names):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif model_type == 'linear' and hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
    else:
        imp = np.zeros(len(feature_names))
    if imp.sum() > 0:
        imp = imp / imp.sum()
    # 确保返回原生 Python float，避免 JSON 序列化问题
    return {feature_names[i]: float(imp[i]) for i in range(len(feature_names))}

def build_training_dataset(ctq_id, product_item=None):
    """构建训练数据集：X 从特征表，Y 优先从覆盖表，否则从 production_data"""
    try:
        feat_query = CtqFeatureValue.query.filter_by(ctq_id=ctq_id)
        if product_item:
            subquery = ProductionData.query.filter_by(product_item=product_item).with_entities(ProductionData.batch_no).subquery()
            feat_query = feat_query.filter(CtqFeatureValue.batch_no.in_(subquery))
        feat_df = pd.read_sql(feat_query.statement, db.engine)
        if feat_df.empty:
            return None, None, None, None

        # 对类别特征进行标签编码
        encodings = {}
        for (feature_name, feature_type), group in feat_df.groupby(['feature_name', 'feature_type']):
            if feature_type == 'categorical':
                unique_vals = group['raw_value'].dropna().unique()
                mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
                encodings[feature_name] = mapping
                group['encoded_value'] = group['raw_value'].map(mapping)
            else:
                group['encoded_value'] = group['feature_value']

        encoded_df = feat_df.copy()
        encoded_df['encoded_value'] = None
        for (feature_name, feature_type), group in feat_df.groupby(['feature_name', 'feature_type']):
            if feature_type == 'categorical':
                mapping = encodings[feature_name]
                encoded_df.loc[group.index, 'encoded_value'] = group['raw_value'].map(mapping)
            else:
                encoded_df.loc[group.index, 'encoded_value'] = group['feature_value']

        pivot_df = encoded_df.pivot(index='batch_no', columns='feature_name', values='encoded_value')
        feature_types_info = {col: feat_df[feat_df['feature_name'] == col]['feature_type'].iloc[0] for col in pivot_df.columns}

        # 将所有列转换为数值类型
        pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce')

        for col in pivot_df.columns:
            if feature_types_info[col] == 'numeric':
                pivot_df[col].fillna(pivot_df[col].median(), inplace=True)
            else:
                pivot_df[col].fillna(-1, inplace=True)

        pivot_df = pivot_df.astype(float)

        batches = pivot_df.index.tolist()
        target_dict = {}
        overrides = CtqTargetOverride.query.filter(
            CtqTargetOverride.batch_no.in_(batches),
            CtqTargetOverride.ctq_id == ctq_id
        ).all()
        for ov in overrides:
            target_dict[ov.batch_no] = ov.target_value

        missing = [b for b in batches if b not in target_dict]
        if missing:
            prod_targets = ProductionData.query.filter(
                ProductionData.batch_no.in_(missing),
                ProductionData.ctq_id == ctq_id
            ).with_entities(ProductionData.batch_no, ProductionData.measured_value).distinct().all()
            for batch, val in prod_targets:
                target_dict[batch] = val

        valid_batches = [b for b in batches if b in target_dict]
        if not valid_batches:
            return None, None, None, None

        X = pivot_df.loc[valid_batches]
        X = X.astype(float)
        y = pd.Series({b: target_dict[b] for b in valid_batches})
        return X, y, feature_types_info, encodings
    except Exception as e:
        print(f"[build_training_dataset] 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def preprocess_data(X, y, missing_method='median', scaler='none'):
    """数据预处理：缺失值填充、标准化，返回处理后的 X, y 以及 scaler 对象（可选）"""
    # 1. 缺失值填充
    if missing_method == 'drop':
        X = X.dropna()
        y = y.loc[X.index]
    else:
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if missing_method == 'median':
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        elif missing_method == 'mean':
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        elif missing_method == 'mode':
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mode().iloc[0])
    # 2. 标准化
    scaler_obj = None
    if scaler != 'none':
        if scaler == 'standard':
            scaler_obj = StandardScaler()
        elif scaler == 'minmax':
            scaler_obj = MinMaxScaler()
        elif scaler == 'robust':
            scaler_obj = RobustScaler()
        if scaler_obj:
            X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns, index=X.index)
    return X, y, scaler_obj

def get_default_param_grid(model_type):
    """返回各模型的默认超参数搜索空间"""
    if model_type == 'lightgbm':
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63]
        }
    elif model_type == 'xgboost':
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'decision_tree':
        return {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5]
        }
    elif model_type == 'linear':
        return {}
    else:
        return {}

def get_model_and_data(ctq_id, product_item, model_type):
    """获取激活的模型、训练数据X、特征列表等（用于高级图表）"""
    # 按训练日期降序取第一条文件存在的模型
    candidates = MlModelMetadata.query.filter_by(
        ctq_id=ctq_id, product_item=product_item, model_type=model_type, is_active=True
    ).order_by(MlModelMetadata.training_date.desc()).all()

    model_meta = None
    for meta in candidates:
        if os.path.exists(meta.model_path):
            model_meta = meta
            break

    if not model_meta:
        return None, None, None, None

    import joblib
    model = joblib.load(model_meta.model_path)
    X, y, _, _ = build_training_dataset(ctq_id, product_item)
    if X is None:
        return None, None, None, None
    feature_names = X.columns.tolist()
    return model, X, y, feature_names

def train_model_task(app, ctq_id, product_item, model_type, hyperparams, task_id, training_config):
    """训练任务（在线程池中执行）"""
    with app.app_context():
        try:
            print(f"[训练任务] 开始, task_id={task_id}, ctq_id={ctq_id}, model={model_type}")
            _tasks[task_id] = {'status': 'loading', 'progress': 10, 'error': None}

            print("步骤1: 构建训练数据集...")
            X, y, feature_types_info, encodings = build_training_dataset(ctq_id, product_item)
            if X is None or len(X) < 50:
                raise ValueError(f'数据不足，当前仅 {len(X) if X is not None else 0} 条记录，至少需要50条')
            _tasks[task_id] = {'status': 'training', 'progress': 20, 'error': None}
            print(f"原始数据: X shape={X.shape}, y size={len(y)}")

            # 预处理
            missing_method = training_config.get('missing_method', 'median')
            scaler = training_config.get('scaler', 'none')
            X, y, scaler_obj = preprocess_data(X, y, missing_method, scaler)
            print(f"预处理后: X shape={X.shape}, y size={len(y)}")

            # 超参数调优配置
            auto_tune = training_config.get('auto_tune', False)
            tune_method = training_config.get('tune_method', 'grid')
            param_grid = training_config.get('param_grid', get_default_param_grid(model_type))
            cv_folds = training_config.get('cv_folds', 5)
            use_cv = training_config.get('use_cv', True)

            ModelClass = get_model_class(model_type)
            best_model = None
            best_params = hyperparams
            best_score = -np.inf
            r2_mean = None
            rmse_mean = None

            if auto_tune and param_grid:
                print("开始自动超参数调优...")
                if tune_method == 'grid':
                    search = GridSearchCV(ModelClass(), param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
                else:
                    search = RandomizedSearchCV(ModelClass(), param_grid, n_iter=20, cv=cv_folds, scoring='r2', n_jobs=-1, random_state=42)
                search.fit(X, y)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_score = search.best_score_
                print(f"最佳参数: {best_params}, 最佳CV R2: {best_score:.4f}")
                _tasks[task_id] = {'status': 'training', 'progress': 50, 'error': None}
            else:
                print("使用指定参数创建模型...")
                best_model = ModelClass(**hyperparams)
                if use_cv:
                    cv_results = cross_validate(best_model, X, y, cv=cv_folds, scoring=('r2', 'neg_mean_squared_error'), return_train_score=False)
                    r2_mean = cv_results['test_r2'].mean()
                    rmse_mean = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
                    best_score = r2_mean
                else:
                    best_model.fit(X, y)
                    y_pred = best_model.predict(X)
                    r2_mean = r2_score(y, y_pred)
                    rmse_mean = np.sqrt(mean_squared_error(y, y_pred))
                _tasks[task_id] = {'status': 'training', 'progress': 60, 'error': None}

            # 最终训练（如果已经拟合过则跳过）
            if not auto_tune and not use_cv:
                pass  # 已经拟合
            elif not auto_tune and use_cv:
                best_model.fit(X, y)  # 使用全部数据重新训练
            # 如果 auto_tune 为 True，best_model 已经拟合

            importance = get_feature_importance(best_model, model_type, X.columns.tolist())

            # 保存模型（文件名使用哈希避免中文路径问题）
            model_dir = os.path.join(app.root_path, 'ml_models')
            os.makedirs(model_dir, exist_ok=True)
            item_key = (product_item or '通用').encode('utf-8')
            safe_item = hashlib.md5(item_key).hexdigest()[:8]
            model_filename = f'ctq_{ctq_id}_{safe_item}_{model_type}_{int(time.time())}.pkl'
            model_path = os.path.join(model_dir, model_filename)

            import joblib
            joblib.dump(best_model, model_path)
            if not os.path.exists(model_path):
                raise IOError(f"模型文件未成功写入：{model_path}")

            # 清理旧模型（保留最新的一个）
            existing = MlModelMetadata.query.filter_by(
                ctq_id=ctq_id, product_item=product_item, model_type=model_type
            ).order_by(MlModelMetadata.training_date.desc()).all()
            for i, m in enumerate(existing):
                if i >= 1:
                    if os.path.exists(m.model_path):
                        os.remove(m.model_path)
                    db.session.delete(m)
            db.session.commit()

            # 保存元数据
            medians_dict = X.median().to_dict()
            for k, v in medians_dict.items():
                medians_dict[k] = float(v)

            metadata = MlModelMetadata(
                ctq_id=ctq_id,
                product_item=product_item,
                model_type=model_type,
                training_samples=len(X),
                model_path=model_path,
                feature_list=json.dumps(X.columns.tolist()),
                importance_json=json.dumps(importance),
                hyperparams=json.dumps(best_params),
                r2_score=float(r2_mean) if r2_mean is not None else float(best_score),
                rmse=float(rmse_mean) if rmse_mean is not None else 0.0,
                is_active=True,
                feature_types=json.dumps(feature_types_info),
                feature_encodings=json.dumps(encodings),
                onehot_columns=None,
                feature_medians=json.dumps(medians_dict),
                training_params=json.dumps(training_config)
            )
            db.session.add(metadata)
            db.session.commit()

            _tasks[task_id] = {
                'status': 'completed',
                'progress': 100,
                'result': {'r2': float(r2_mean) if r2_mean is not None else float(best_score), 'rmse': float(rmse_mean) if rmse_mean is not None else 0.0, 'importance': importance, 'best_params': best_params},
                'error': None
            }
            app.logger.info(f"CTQ {ctq_id} 模型 {model_type} 训练完成，R2={r2_mean if r2_mean is not None else best_score:.4f}")
        except Exception as e:
            app.logger.exception("训练失败")
            print(f"[训练任务] 异常: {e}")
            import traceback
            traceback.print_exc()
            _tasks[task_id] = {'status': 'failed', 'progress': 0, 'error': str(e)}

# ---------- 路由 ----------
@influence_bp.route('/analysis')
def analysis_page():
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return "影响因素分析模块未启用", 403
    ctq_id = request.args.get('ctq_id', type=int)
    product_item = request.args.get('product_item', '')
    ctq = CTQConfig.query.get_or_404(ctq_id)
    return render_template('influence_analysis.html', ctq=ctq, product_item=product_item)

@influence_bp.route('/api/feature_upload', methods=['POST'])
def upload_features():
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    if 'file' not in request.files:
        return jsonify({'error': '请选择文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型，请上传 .xlsx, .xls, .csv'}), 400
    if request.content_length > MAX_FILE_SIZE:
        return jsonify({'error': f'文件大小超过 {MAX_FILE_SIZE // (1024*1024)}MB 限制'}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        return jsonify({'error': f'文件解析失败: {str(e)}'}), 400

    required = ['batch_no', 'ctq_name', 'feature_name', 'feature_value']
    if not all(c in df.columns for c in required):
        return jsonify({'error': f'缺少必需列：{required}'}), 400

    has_target = 'target_value' in df.columns
    has_feature_type = 'feature_type' in df.columns
    errors = []
    success = 0
    target_updates = {}

    for idx, row in df.iterrows():
        line_num = idx + 2
        batch_no = str(row['batch_no']).strip()
        ctq_name = str(row['ctq_name']).strip()
        feature_name = str(row['feature_name']).strip()
        raw_value = row['feature_value']

        if has_feature_type and pd.notna(row['feature_type']):
            feature_type = str(row['feature_type']).strip().lower()
            if feature_type not in ('numeric', 'categorical'):
                errors.append(f"第{line_num}行：feature_type 只能是 numeric 或 categorical")
                continue
        else:
            try:
                float(raw_value)
                feature_type = 'numeric'
            except:
                feature_type = 'categorical'

        if feature_type == 'numeric':
            try:
                feature_value = float(raw_value)
                raw_value_saved = None
            except:
                errors.append(f"第{line_num}行：numeric 类型但值不是有效数字")
                continue
        else:
            feature_value = None
            raw_value_saved = str(raw_value)

        ctq = CTQConfig.query.filter_by(ctq_name=ctq_name).first()
        if not ctq:
            errors.append(f"第{line_num}行：CTQ '{ctq_name}' 不存在")
            continue
        prod = ProductionData.query.filter_by(batch_no=batch_no).first()
        if not prod:
            errors.append(f"第{line_num}行：批次 '{batch_no}' 不存在")
            continue

        if has_target and pd.notna(row['target_value']):
            try:
                target_val = float(row['target_value'])
                target_updates[(batch_no, ctq.ctq_id)] = target_val
            except:
                errors.append(f"第{line_num}行：target_value 不是有效数字")
                continue

        existing = CtqFeatureValue.query.filter_by(
            batch_no=batch_no, ctq_id=ctq.ctq_id, feature_name=feature_name
        ).first()
        if existing:
            existing.feature_value = feature_value
            existing.raw_value = raw_value_saved
            existing.feature_type = feature_type
        else:
            new_feat = CtqFeatureValue(
                batch_no=batch_no,
                ctq_id=ctq.ctq_id,
                feature_name=feature_name,
                feature_value=feature_value,
                raw_value=raw_value_saved,
                feature_type=feature_type
            )
            db.session.add(new_feat)
        success += 1

    for (batch_no, ctq_id), target_val in target_updates.items():
        existing = CtqTargetOverride.query.get((batch_no, ctq_id))
        if existing:
            existing.target_value = target_val
            existing.source_file = secure_filename(file.filename)
            existing.update_time = datetime.now()
        else:
            new_target = CtqTargetOverride(
                batch_no=batch_no,
                ctq_id=ctq_id,
                target_value=target_val,
                source_file=secure_filename(file.filename)
            )
            db.session.add(new_target)

    if errors:
        db.session.rollback()
        return jsonify({'error': '上传失败', 'details': errors[:20]}), 400

    db.session.commit()
    msg = f'成功处理 {success} 条特征记录'
    if target_updates:
        msg += f'，并更新 {len(target_updates)} 个批次的自定义目标值'
    return jsonify({'message': msg})

@influence_bp.route('/api/features/<int:ctq_id>')
def get_features(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    query = CtqFeatureValue.query.filter_by(ctq_id=ctq_id)
    if product_item:
        subquery = ProductionData.query.filter_by(product_item=product_item).with_entities(ProductionData.batch_no).subquery()
        query = query.filter(CtqFeatureValue.batch_no.in_(subquery))
    from sqlalchemy import func
    stats = db.session.query(
        CtqFeatureValue.feature_name,
        CtqFeatureValue.feature_type,
        func.count(func.distinct(CtqFeatureValue.batch_no)).label('batch_count')
    ).filter(CtqFeatureValue.ctq_id == ctq_id)
    if product_item:
        stats = stats.filter(CtqFeatureValue.batch_no.in_(subquery))
    stats = stats.group_by(CtqFeatureValue.feature_name, CtqFeatureValue.feature_type).all()
    features = [{'name': f, 'type': t, 'batch_count': c} for f, t, c in stats]
    return jsonify({'features': features})

@influence_bp.route('/api/feature_type/<int:ctq_id>', methods=['POST'])
def set_feature_type(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    data = request.get_json()
    feature_name = data.get('feature_name')
    new_type = data.get('feature_type')
    if new_type not in ('numeric', 'categorical'):
        return jsonify({'error': '无效类型'}), 400
    CtqFeatureValue.query.filter_by(ctq_id=ctq_id, feature_name=feature_name).update({'feature_type': new_type})
    db.session.commit()
    return jsonify({'success': True})

@influence_bp.route('/api/model_types')
def list_model_types():
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    types = MlModelType.query.filter_by(is_active=True).all()
    return jsonify([{
        'key': t.model_key,
        'name': t.display_name,
        'default_params': json.loads(t.default_params) if t.default_params else {}
    } for t in types])

@influence_bp.route('/api/train/<int:ctq_id>', methods=['POST'])
def train(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求体不能为空'}), 400
    product_item = normalize_product_item(data.get('product_item'))
    model_type = data.get('model_type', 'lightgbm')
    hyperparams = data.get('hyperparams', {})
    training_config = data.get('training_config', {
        'missing_method': 'median',
        'scaler': 'none',
        'use_cv': True,
        'cv_folds': 5,
        'auto_tune': False,
        'tune_method': 'grid',
        'param_grid': get_default_param_grid(model_type)
    })
    X, y, _, _ = build_training_dataset(ctq_id, product_item)
    if X is None or len(X) < 50:
        return jsonify({'error': f'数据不足，当前仅 {len(X) if X is not None else 0} 条记录，至少需要50条'}), 400
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {'status': 'queued', 'progress': 0, 'error': None}
    app = current_app._get_current_object()
    _executor.submit(train_model_task, app, ctq_id, product_item, model_type, hyperparams, task_id, training_config)
    return jsonify({'task_id': task_id, 'message': '训练任务已启动'})

@influence_bp.route('/api/train_status/<task_id>')
def train_status(task_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    info = _tasks.get(task_id)
    if not info:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(info)

@influence_bp.route('/api/models/<int:ctq_id>')
def get_models(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    models = MlModelMetadata.query.filter_by(ctq_id=ctq_id, product_item=product_item)\
        .order_by(MlModelMetadata.training_date.desc()).all()
    latest = {}
    for m in models:
        if m.model_type not in latest:
            latest[m.model_type] = m
    result = []
    for m in latest.values():
        result.append({
            'model_type': m.model_type,
            'training_date': m.training_date.isoformat(),
            'training_samples': m.training_samples,
            'r2': m.r2_score,
            'rmse': m.rmse,
            'is_active': m.is_active,
            'importance': json.loads(m.importance_json) if m.importance_json else {}
        })
    return jsonify(result)

@influence_bp.route('/api/importance/<int:ctq_id>/<model_type>')
def get_importance(ctq_id, model_type):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    model_meta = MlModelMetadata.query.filter_by(
        ctq_id=ctq_id, product_item=product_item, model_type=model_type, is_active=True
    ).first()
    if not model_meta:
        return jsonify({'error': '模型不存在或未训练'}), 404
    importance = json.loads(model_meta.importance_json)
    return jsonify({
        'features': list(importance.keys()),
        'scores': list(importance.values())
    })

# ---------- 高级图表 API ----------
@influence_bp.route('/api/shap/<int:ctq_id>/<model_type>')
def shap_summary(ctq_id, model_type):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    model, X, y, feature_names = get_model_and_data(ctq_id, product_item, model_type)
    if model is None:
        return jsonify({'error': '模型不存在或数据缺失'}), 404
    import shap
    if model_type in ('lightgbm', 'xgboost', 'random_forest', 'decision_tree'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
    else:
        return jsonify({'error': f'不支持的模型类型 {model_type}'}), 400
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    shap_data = [{'feature': feature_names[idx], 'values': shap_values[:, idx].tolist()} for idx in sorted_idx]
    base_value = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
    return jsonify({'shap_data': shap_data, 'base_value': base_value})

@influence_bp.route('/api/pdp/<int:ctq_id>/<model_type>')
def partial_dependence(ctq_id, model_type):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    feature_name = request.args.get('feature_name', '')
    if not feature_name:
        return jsonify({'error': '缺少 feature_name'}), 400
    model, X, y, feature_names = get_model_and_data(ctq_id, product_item, model_type)
    if model is None or feature_name not in feature_names:
        return jsonify({'error': '模型或特征不存在'}), 404
    model_meta = MlModelMetadata.query.filter_by(
        ctq_id=ctq_id, product_item=product_item, model_type=model_type, is_active=True
    ).first()
    feature_types = json.loads(model_meta.feature_types) if model_meta.feature_types else {}
    is_categorical = feature_types.get(feature_name) == 'categorical'
    if is_categorical:
        grid = np.unique(X[feature_name]).tolist()
    else:
        min_val, max_val = X[feature_name].min(), X[feature_name].max()
        grid = np.linspace(min_val, max_val, 50).tolist()
    X_temp = X.copy()
    y_preds = []
    for val in grid:
        X_temp[feature_name] = val
        y_preds.append(float(model.predict(X_temp).mean()))
    return jsonify({
        'feature': feature_name,
        'x': grid,
        'y': y_preds,
        'is_categorical': is_categorical
    })

@influence_bp.route('/api/diagnostics/<int:ctq_id>/<model_type>')
def diagnostics(ctq_id, model_type):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    model, X, y, _ = get_model_and_data(ctq_id, product_item, model_type)
    if model is None:
        return jsonify({'error': '模型不存在'}), 404
    y_pred = model.predict(X)
    residuals = y - y_pred
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return jsonify({
        'actual': y.tolist(),
        'predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'r2': r2,
        'rmse': rmse
    })

@influence_bp.route('/api/correlation/<int:ctq_id>')
def feature_correlation(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    X, _, _, _ = build_training_dataset(ctq_id, product_item)
    if X is None:
        return jsonify({'error': '无数据'}), 400
    corr = X.corr().round(4).values.tolist()
    features = X.columns.tolist()
    return jsonify({'features': features, 'correlation': corr})

# ========== 新增清空特征数据路由 ==========
@influence_bp.route('/api/clear_features/<int:ctq_id>', methods=['POST'])
def clear_features(ctq_id):
    """清空当前CTQ及品项下的所有特征数据"""
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403

    product_item = request.args.get('product_item', '')
    query = CtqFeatureValue.query.filter_by(ctq_id=ctq_id)

    if product_item:
        # 获取该品项下的所有批次号
        subquery = db.session.query(ProductionData.batch_no).filter(
            ProductionData.product_item == product_item
        ).subquery()
        query = query.filter(CtqFeatureValue.batch_no.in_(subquery))

    deleted_count = query.delete(synchronize_session=False)
    db.session.commit()
    return jsonify({'success': True, 'deleted': deleted_count, 'message': f'已清空 {deleted_count} 条特征记录'})

# 预留预测接口
@influence_bp.route('/api/predict/<int:ctq_id>', methods=['POST'])
def predict(ctq_id):
    return jsonify({'error': '预测功能开发中'}), 501