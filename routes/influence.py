# routes/influence.py
import json
import os
import uuid
import time
import hashlib
import joblib
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from sklearn.model_selection import (
    cross_validate, KFold, GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from models.database import db
from models.models import CTQConfig, ProductionData
from models.influence_models import (
    CtqFeatureValue, CtqTargetOverride, MlModelMetadata, MlModelType, SystemSetting
)
from utils import normalize_product_item

influence_bp = Blueprint('influence', __name__, url_prefix='/influence')

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

_executor = ThreadPoolExecutor(max_workers=2)
_tasks = {}
_tasks_lock = Lock()


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
        if imp.ndim > 1:
            imp = imp[0]
    else:
        imp = np.zeros(len(feature_names))
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return {feature_names[i]: float(imp[i]) for i in range(len(feature_names))}


def build_training_dataset(ctq_id, product_item=None):
    """构建训练数据集：X 保留原始值，Y 优先覆盖值"""
    try:
        feat_query = CtqFeatureValue.query.filter_by(ctq_id=ctq_id)
        if product_item:
            subquery = ProductionData.query.filter_by(product_item=product_item).with_entities(
                ProductionData.batch_no).subquery()
            feat_query = feat_query.filter(CtqFeatureValue.batch_no.in_(subquery))
        feat_df = pd.read_sql(feat_query.statement, db.engine)
        if feat_df.empty:
            return None, None, None, None

        feat_df['raw_or_feature'] = feat_df.apply(
            lambda row: row['raw_value'] if row['feature_type'] == 'categorical' else row['feature_value'], axis=1
        )

        pivot_df = feat_df.pivot(index='batch_no', columns='feature_name', values='raw_or_feature')
        feature_types_info = {col: feat_df[feat_df['feature_name'] == col]['feature_type'].iloc[0] for col in
                              pivot_df.columns}

        for col, ftype in feature_types_info.items():
            if ftype == 'numeric':
                pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce')
            else:
                pivot_df[col] = pivot_df[col].fillna('missing')

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
            prod_query = db.session.query(
                ProductionData.batch_no,
                db.func.avg(ProductionData.measured_value).label('mean_value')
            ).filter(
                ProductionData.batch_no.in_(missing),
                ProductionData.ctq_id == ctq_id
            )
            if product_item:
                prod_query = prod_query.filter(ProductionData.product_item == product_item)
            prod_targets = prod_query.group_by(ProductionData.batch_no).all()
            for batch, val in prod_targets:
                target_dict[batch] = val

        valid_batches = [b for b in batches if b in target_dict]
        if not valid_batches:
            return None, None, None, None

        X = pivot_df.loc[valid_batches]
        y = pd.Series({b: target_dict[b] for b in valid_batches})
        X = X.sort_index()
        y = y.sort_index()
        return X, y, feature_types_info, None
    except Exception as e:
        print(f"[build_training_dataset] 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def build_preprocessing_pipeline(feature_types_info, training_config):
    numeric_features = [col for col, ftype in feature_types_info.items() if ftype == 'numeric']
    categorical_features = [col for col, ftype in feature_types_info.items() if ftype == 'categorical']
    transformers = []

    if numeric_features:
        num_steps = []
        missing_method = training_config.get('missing_method', 'median')
        strategy_map = {'median': 'median', 'mean': 'mean', 'mode': 'most_frequent'}
        strategy = strategy_map.get(missing_method, 'median')
        num_steps.append(('imputer', SimpleImputer(strategy=strategy)))

        scaler_name = training_config.get('scaler', 'none')
        if scaler_name == 'standard':
            num_steps.append(('scaler', StandardScaler()))
        elif scaler_name == 'minmax':
            num_steps.append(('scaler', MinMaxScaler()))
        elif scaler_name == 'robust':
            num_steps.append(('scaler', RobustScaler()))

        transformers.append(('num', Pipeline(num_steps), numeric_features))

    if categorical_features:
        cat_steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]
        transformers.append(('cat', Pipeline(cat_steps), categorical_features))

    return ColumnTransformer(transformers, remainder='drop')


def get_model_and_data(ctq_id, product_item, model_type):
    """加载最新活跃模型 Pipeline，兼容旧版裸模型"""
    candidates = MlModelMetadata.query.filter_by(
        ctq_id=ctq_id, product_item=product_item, model_type=model_type, is_active=True
    ).order_by(MlModelMetadata.training_date.desc()).all()

    for meta in candidates:
        if not os.path.exists(meta.model_path):
            continue
        pipeline = joblib.load(meta.model_path)

        if not isinstance(pipeline, Pipeline):
            current_app.logger.warning(
                f"模型 {meta.id} 为旧版格式（裸模型），无法用于解释分析，请重新训练"
            )
            continue

        X_orig, y, _, _ = build_training_dataset(ctq_id, product_item)
        if X_orig is None:
            continue

        try:
            preprocessor = pipeline.named_steps['preprocessor']
            X_preprocessed = preprocessor.transform(X_orig)
            feature_names = X_orig.columns.tolist()
            return pipeline, X_orig, X_preprocessed, y, feature_names
        except Exception as e:
            current_app.logger.error(f"模型预处理失败: {e}")
            continue

    return None, None, None, None, None


def get_default_param_grid(model_type):
    if model_type == 'lightgbm':
        return {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [15, 31, 63]}
    elif model_type == 'xgboost':
        return {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
    elif model_type == 'random_forest':
        return {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
    elif model_type == 'decision_tree':
        return {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5]}
    elif model_type == 'linear':
        return {}
    else:
        return {}


def train_model_task(app, ctq_id, product_item, model_type, hyperparams, task_id, training_config):
    with app.app_context():
        try:
            def update_task(status, progress, error=None, result=None):
                with _tasks_lock:
                    _tasks[task_id] = {'status': status, 'progress': progress, 'error': error}
                    if result:
                        _tasks[task_id]['result'] = result

            print(f"\n[训练任务] 开始, task_id={task_id}, ctq_id={ctq_id}, model={model_type}", flush=True)
            update_task('loading', 10)

            X, y, feature_types_info, _ = build_training_dataset(ctq_id, product_item)
            if X is None or len(X) < 10:
                raise ValueError(f'数据不足，当前仅 {len(X) if X is not None else 0} 条记录，至少需要10条')

            # 诊断日志
            print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}", flush=True)
            print(f"[DEBUG] y describe:\n{y.describe()}", flush=True)
            print(f"[DEBUG] y 前5个值: {y.head().tolist()}", flush=True)
            print(f"[DEBUG] X 前3行:\n{X.head(3)}", flush=True)
            print(f"[DEBUG] X index == y index: {X.index.equals(y.index)}", flush=True)

            # ********** 新增：特征与目标的相关性 **********
            num_X = X.select_dtypes(include=[np.number])
            if not num_X.empty:
                corr_series = num_X.corrwith(y).sort_values(ascending=False)
                print("[DEBUG] 特征与目标的皮尔逊相关系数：")
                for feat, corr_val in corr_series.items():
                    print(f"  {feat}: {corr_val:.4f}", flush=True)

            # 离群值过滤（可调整倍数，这里保持1.5，如有需要手动改为2.0或3.0）
            IQR_multiplier = training_config.get('iqr_multiplier', 1.5)
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (y < (Q1 - IQR_multiplier * IQR)) | (y > (Q3 + IQR_multiplier * IQR))
            if outlier_mask.any():
                print(f"[DEBUG] 检测到 {outlier_mask.sum()} 个离群值 (IQR×{IQR_multiplier})，将被移除", flush=True)
                y = y[~outlier_mask]
                X = X.loc[y.index]
                if len(X) < 10:
                    raise ValueError('离群值过滤后数据不足，无法训练')

            # 可选目标变换（可在高级参数中启用，此处作为调试开关）
            apply_log = training_config.get('log_transform', False)
            if apply_log:
                y = np.log1p(y)
                print("[DEBUG] 已对目标值应用 log1p 变换", flush=True)

            update_task('training', 20)

            preprocessor = build_preprocessing_pipeline(feature_types_info, training_config)
            temp_prep = Pipeline([('preprocessor', preprocessor)])
            X_preview = temp_prep.fit_transform(X)
            print(f"[DEBUG] 预处理后 X 形状: {X_preview.shape}", flush=True)
            print(f"[DEBUG] 预处理后 X 每列方差（前10列）: {np.nanvar(X_preview, axis=0).round(4)[:10]}", flush=True)

            ModelClass = get_model_class(model_type)
            model = ModelClass(**hyperparams)
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            auto_tune = training_config.get('auto_tune', False)
            tune_method = training_config.get('tune_method', 'grid')
            param_grid = training_config.get('param_grid', get_default_param_grid(model_type))
            cv_folds = training_config.get('cv_folds', 5)
            use_cv = training_config.get('use_cv', True)

            best_params = hyperparams
            r2_mean = None
            rmse_mean = None

            if auto_tune and param_grid:
                param_grid_prefix = {f'model__{k}': v for k, v in param_grid.items()}
                print("开始自动超参数调优...", flush=True)
                if tune_method == 'grid':
                    search = GridSearchCV(full_pipeline, param_grid_prefix, cv=cv_folds, scoring='r2', n_jobs=1)
                else:
                    search = RandomizedSearchCV(full_pipeline, param_grid_prefix, n_iter=20, cv=cv_folds, scoring='r2',
                                                n_jobs=1, random_state=42)
                search.fit(X, y)
                full_pipeline = search.best_estimator_
                best_params = search.best_params_
                r2_mean = search.best_score_
                print(f"最佳参数: {best_params}, 最佳CV R2: {r2_mean:.4f}", flush=True)
                update_task('training', 50)
            else:
                if use_cv:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_results = cross_validate(
                        full_pipeline, X, y,
                        cv=cv,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=False,
                        return_estimator=True
                    )
                    r2_scores = cv_results['test_r2']
                    print(f"[DEBUG] CV 各折 R²: {r2_scores}", flush=True)

                    fold0_idx = list(cv.split(X, y))[0]
                    X_test_fold = X.iloc[fold0_idx[1]]
                    y_test_fold = y.iloc[fold0_idx[1]]
                    y_pred_fold = cv_results['estimator'][0].predict(X_test_fold)
                    print(f"[DEBUG] 第一折真实值前10: {y_test_fold.head(10).tolist()}", flush=True)
                    print(f"[DEBUG] 第一折预测值前10: {y_pred_fold[:10].tolist()}", flush=True)

                    r2_mean = r2_scores.mean()
                    rmse_mean = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
                else:
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                    full_pipeline.fit(X_train, y_train)
                    y_pred = full_pipeline.predict(X_val)
                    r2_mean = r2_score(y_val, y_pred)
                    rmse_mean = np.sqrt(mean_squared_error(y_val, y_pred))
                    print(f"[DEBUG] 验证集 R²: {r2_mean:.4f}, RMSE: {rmse_mean:.2f}", flush=True)
                    print(f"[DEBUG] 验证集真实值前5: {y_val.head().tolist()}", flush=True)
                    print(f"[DEBUG] 验证集预测值前5: {y_pred[:5].tolist()}", flush=True)
                update_task('training', 60)

            # 在全量数据上重新训练
            full_pipeline.fit(X, y)

            final_model = full_pipeline.named_steps['model']
            importance = get_feature_importance(final_model, model_type, X.columns.tolist())

            # 保存模型
            model_dir = os.path.join(app.root_path, 'ml_models')
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f'ctq_{ctq_id}_{product_item[:8]}_{uuid.uuid4().hex[:8]}.pkl'
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(full_pipeline, model_path)

            # 更新数据库：旧模型置为非活跃
            MlModelMetadata.query.filter_by(
                ctq_id=ctq_id, product_item=product_item, model_type=model_type
            ).update({'is_active': False})
            old_models = MlModelMetadata.query.filter_by(
                ctq_id=ctq_id, product_item=product_item, model_type=model_type, is_active=False
            ).all()
            for old in old_models:
                if os.path.exists(old.model_path):
                    try:
                        os.remove(old.model_path)
                    except OSError:
                        pass
            db.session.commit()

            medians_dict = X.median(numeric_only=True).to_dict()
            for k, v in medians_dict.items():
                if isinstance(v, (np.floating, np.integer)):
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
                r2_score=float(r2_mean) if r2_mean is not None else 0.0,
                rmse=float(rmse_mean) if rmse_mean is not None else 0.0,
                is_active=True,
                feature_types=json.dumps(feature_types_info),
                feature_encodings=json.dumps({}),
                onehot_columns=None,
                feature_medians=json.dumps(medians_dict),
                training_params=json.dumps(training_config)
            )
            db.session.add(metadata)
            db.session.commit()

            update_task('completed', 100, result={
                'r2': float(r2_mean) if r2_mean is not None else 0.0,
                'rmse': float(rmse_mean) if rmse_mean is not None else 0.0,
                'importance': importance,
                'best_params': best_params
            })
            print(f"CTQ {ctq_id} 模型 {model_type} 训练完成，R2={r2_mean:.4f}", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            with _tasks_lock:
                _tasks[task_id] = {'status': 'failed', 'progress': 0, 'error': str(e)}


# ========== 以下路由与之前完全一致，为节省篇幅只列出重要部分 ==========
# 请将前面回答中完整的路由部分复制到此（包括 analysis_page, upload_features, get_features, set_feature_type,
# list_model_types, train, train_status, get_models, get_importance, shap_summary,
# partial_dependence, diagnostics, feature_correlation, clear_features, settings 等）
# 注意确保所有路由函数定义完整。


# ========== 路由 ==========

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
        return jsonify({'error': f'文件大小超过 {MAX_FILE_SIZE // (1024 * 1024)}MB 限制'}), 400

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
        subquery = ProductionData.query.filter_by(product_item=product_item).with_entities(
            ProductionData.batch_no).subquery()
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
    if X is None or len(X) < 10:
        return jsonify(
            {'error': f'数据不足，当前仅 {len(X) if X is not None else 0} 条记录，至少需要10条'}), 400
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {'status': 'queued', 'progress': 0, 'error': None}
    app = current_app._get_current_object()
    _executor.submit(train_model_task, app, ctq_id, product_item, model_type, hyperparams, task_id, training_config)
    return jsonify({'task_id': task_id, 'message': '训练任务已启动'})


@influence_bp.route('/api/train_status/<task_id>')
def train_status(task_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    with _tasks_lock:
        info = _tasks.get(task_id)
    if not info:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(info)


@influence_bp.route('/api/models/<int:ctq_id>')
def get_models(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    models = MlModelMetadata.query.filter_by(ctq_id=ctq_id, product_item=product_item) \
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


@influence_bp.route('/api/shap/<int:ctq_id>/<model_type>')
def shap_summary(ctq_id, model_type):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    pipeline, X_orig, X_preprocessed, y, feature_names = get_model_and_data(ctq_id, product_item, model_type)
    if pipeline is None:
        return jsonify({'error': '没有可用的有效模型，请重新训练模型后再试'}), 404
    import shap
    model = pipeline.named_steps['model']
    if model_type in ('lightgbm', 'xgboost', 'random_forest', 'decision_tree'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_preprocessed)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_preprocessed)
        shap_values = explainer.shap_values(X_preprocessed)
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
    pipeline, X_orig, X_preprocessed, y, feature_names = get_model_and_data(ctq_id, product_item, model_type)
    if pipeline is None:
        return jsonify({'error': '模型或特征不存在'}), 404

    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']

    encoder = None
    for name, transformer, cols in preprocessor.transformers_:
        if feature_name in cols:
            if name == 'cat' and 'encoder' in transformer.named_steps:
                encoder = transformer.named_steps['encoder']
            break

    X_temp = X_orig.copy()
    if encoder:
        cat_idx = list(encoder.feature_names_in_).index(feature_name) if hasattr(encoder, 'feature_names_in_') else 0
        categories = encoder.categories_[cat_idx]
        grid = list(categories)
        preds = []
        for val in grid:
            X_temp[feature_name] = val
            X_processed = preprocessor.transform(X_temp)
            preds.append(float(model.predict(X_processed).mean()))
        return jsonify({'feature': feature_name, 'x': grid, 'y': preds, 'is_categorical': True})
    else:
        min_val = X_orig[feature_name].min()
        max_val = X_orig[feature_name].max()
        grid = np.linspace(min_val, max_val, 50)
        preds = []
        for val in grid:
            X_temp[feature_name] = val
            X_processed = preprocessor.transform(X_temp)
            preds.append(float(model.predict(X_processed).mean()))
        return jsonify({'feature': feature_name, 'x': grid.tolist(), 'y': preds, 'is_categorical': False})


@influence_bp.route('/api/diagnostics/<int:ctq_id>/<model_type>')
def diagnostics(ctq_id, model_type):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    pipeline, X_orig, X_preprocessed, y, feature_names = get_model_and_data(ctq_id, product_item, model_type)
    if pipeline is None:
        return jsonify({'error': '模型不存在'}), 404
    model = pipeline.named_steps['model']
    y_pred = model.predict(X_preprocessed)
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
    num_X = X.select_dtypes(include=[np.number])
    if num_X.shape[1] < 2:
        return jsonify({'features': num_X.columns.tolist(), 'correlation': []})
    corr = num_X.corr().round(4).values.tolist()
    features = num_X.columns.tolist()
    return jsonify({'features': features, 'correlation': corr})


@influence_bp.route('/api/clear_features/<int:ctq_id>', methods=['POST'])
def clear_features(ctq_id):
    if not current_app.config.get('ENABLE_ML_INFLUENCE', False):
        return jsonify({'error': '模块未启用'}), 403
    product_item = request.args.get('product_item', '')
    query = CtqFeatureValue.query.filter_by(ctq_id=ctq_id)
    if product_item:
        subquery = db.session.query(ProductionData.batch_no).filter(
            ProductionData.product_item == product_item
        ).subquery()
        query = query.filter(CtqFeatureValue.batch_no.in_(subquery))
    deleted_count = query.delete(synchronize_session=False)
    db.session.commit()
    return jsonify({'success': True, 'deleted': deleted_count, 'message': f'已清空 {deleted_count} 条特征记录'})


@influence_bp.route('/api/settings/ml_influence', methods=['GET'])
def get_ml_influence_status():
    setting = SystemSetting.query.filter_by(key='ENABLE_ML_INFLUENCE').first()
    enabled = True
    if setting:
        enabled = setting.value.lower() == 'true'
    current_app.config['ENABLE_ML_INFLUENCE'] = enabled
    return jsonify({'enabled': enabled})


@influence_bp.route('/api/settings/ml_influence', methods=['POST'])
def set_ml_influence_status():
    data = request.get_json()
    if not data or 'enabled' not in data:
        return jsonify({'error': '缺少 enabled 字段'}), 400
    enabled = bool(data['enabled'])

    setting = SystemSetting.query.filter_by(key='ENABLE_ML_INFLUENCE').first()
    if setting:
        setting.value = str(enabled).lower()
    else:
        setting = SystemSetting(key='ENABLE_ML_INFLUENCE', value=str(enabled).lower())
        db.session.add(setting)
    db.session.commit()

    current_app.config['ENABLE_ML_INFLUENCE'] = enabled
    return jsonify({'enabled': enabled, 'message': '设置已更新'})


@influence_bp.route('/api/predict/<int:ctq_id>', methods=['POST'])
def predict(ctq_id):
    return jsonify({'error': '预测功能开发中'}), 501