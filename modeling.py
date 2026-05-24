import os
import io
import base64
import json
import traceback
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, OrdinalEncoder,
                                   MinMaxScaler, RobustScaler, FunctionTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import optuna
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
import shap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import joblib

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


class DataPreprocessor:
    """智能数据预处理器，支持数值型和字符串离散特征"""
    def __init__(self, config):
        self.config = config
        self.continuous_features = []
        self.discrete_features = []
        self.preprocessor = None
        self.feature_names_out = []

    def detect_discrete_candidates(self, X):
        candidates = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                n_unique = X[col].nunique()
                if n_unique <= self.config.get('discrete_max_cardinality', 15):
                    candidates[col] = int(n_unique)
            elif pd.api.types.is_string_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
                n_unique = X[col].nunique()
                if n_unique <= self.config.get('discrete_max_cardinality', 15):
                    candidates[col] = int(n_unique)
        return candidates

    def build_preprocessor(self, X):
        all_cols = X.columns.tolist()
        manual_discrete = self.config.get('discrete_columns', [])
        if self.config.get('auto_detect_discrete', True):
            auto_candidates = self.detect_discrete_candidates(X)
            for col in auto_candidates:
                if col not in manual_discrete and col in all_cols:
                    manual_discrete.append(col)
        discrete_cols = [c for c in manual_discrete if c in all_cols]
        continuous_candidates = [c for c in all_cols if c not in discrete_cols]
        continuous_cols = [c for c in continuous_candidates if pd.api.types.is_numeric_dtype(X[c])]
        self.discrete_features = discrete_cols
        self.continuous_features = continuous_cols

        transformers = []
        cont_steps = []
        missing_strategy = self.config.get('missing_strategy', 'median')
        if missing_strategy != 'none':
            if missing_strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif missing_strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif missing_strategy == 'constant':
                fill_val = self.config.get('missing_fill_value', 0)
                imputer = SimpleImputer(strategy='constant', fill_value=fill_val)
            else:
                imputer = SimpleImputer(strategy='median')
            cont_steps.append(('imputer', imputer))
        outlier_method = self.config.get('outlier_method', 'none')
        if outlier_method == 'iqr':
            cont_steps.append(('outlier', FunctionTransformer(self._iqr_clip, validate=False)))
        elif outlier_method == 'zscore':
            cont_steps.append(('outlier', FunctionTransformer(self._zscore_clip, validate=False)))
        scaler_method = self.config.get('scaler', 'standard')
        if scaler_method != 'none':
            if scaler_method == 'standard':
                scaler = StandardScaler()
            elif scaler_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            cont_steps.append(('scaler', scaler))
        cont_pipe = Pipeline(cont_steps) if cont_steps else 'passthrough'
        if continuous_cols:
            transformers.append(('cont', cont_pipe, continuous_cols))

        if discrete_cols:
            disc_steps = []
            disc_imputer = SimpleImputer(strategy='most_frequent')
            disc_steps.append(('imputer', disc_imputer))
            encode_method = self.config.get('discrete_encode', 'onehot')
            if encode_method == 'onehot':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                disc_steps.append(('encoder', encoder))
            elif encode_method == 'ordinal':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                disc_steps.append(('encoder', encoder))
            disc_pipe = Pipeline(disc_steps)
            transformers.append(('disc', disc_pipe, discrete_cols))

        self.preprocessor = ColumnTransformer(transformers, remainder='drop')
        return self.preprocessor

    def fit_transform(self, X, y=None):
        if self.preprocessor is None:
            self.build_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names_out = self._get_feature_names(self.preprocessor)
        return pd.DataFrame(X_processed, columns=self.feature_names_out)

    def transform(self, X):
        X_processed = self.preprocessor.transform(X)
        return pd.DataFrame(X_processed, columns=self.feature_names_out)

    def _get_feature_names(self, column_transformer):
        names = []
        for name, trans, cols in column_transformer.transformers_:
            if name == 'remainder':
                continue
            if trans == 'passthrough':
                names.extend(cols)
            elif hasattr(trans, 'get_feature_names_out'):
                names.extend(trans.get_feature_names_out(cols))
            elif isinstance(trans, Pipeline):
                last_step = trans.steps[-1][1] if trans.steps else None
                if hasattr(last_step, 'get_feature_names_out'):
                    names.extend(last_step.get_feature_names_out(cols))
                else:
                    if isinstance(last_step, OrdinalEncoder):
                        names.extend(cols)
                    else:
                        names.extend(cols)
            else:
                names.extend(cols)
        return names

    def _iqr_clip(self, X):
        X = np.array(X, dtype=float)
        for i in range(X.shape[1]):
            col = X[:, i]
            q1 = np.nanpercentile(col, 25)
            q3 = np.nanpercentile(col, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            col = np.clip(col, lower, upper)
            X[:, i] = col
        return X

    def _zscore_clip(self, X, threshold=3):
        X = np.array(X, dtype=float)
        for i in range(X.shape[1]):
            col = X[:, i]
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std == 0:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
            col = np.clip(col, lower, upper)
            X[:, i] = col
        return X


def run_full_pipeline(config, progress_callback=None):
    result = {'status': 'success', 'message': ''}
    task_dir = config.get('task_dir', '')

    def save_fig_to_task(fig, basename):
        if not task_dir:
            return fig_to_base64(fig)
        path = os.path.join(task_dir, basename)
        fig.savefig(path, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        task_id = os.path.basename(task_dir)
        return f'/ml-tool/static/{task_id}/{basename}'

    def log(msg, inc=1):
        if progress_callback:
            progress_callback(msg, inc)

    try:
        log("📂 加载数据...")
        df = pd.read_csv(config['data_path'])
        drop_cols = [c.strip() for c in config['drop_cols'].split(',') if c.strip()]
        if config['target_col'] in drop_cols:
            raise ValueError(f'目标列 "{config["target_col"]}" 不能出现在删除列中')
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        if config['target_col'] not in df.columns:
            return {'status': 'error', 'message': f'目标列 "{config["target_col"]}" 不存在'}

        y = df[config['target_col']]
        X = df.drop(columns=[config['target_col']])
        result['original_shape'] = f'{X.shape[0]}样本, {X.shape[1]}个原始特征'

        # 保存原始特征数据，供随机样本使用
        original_data_path = os.path.join(task_dir, 'original_data.pkl')
        X.to_pickle(original_data_path)

        log("✂️ 划分训练/测试集...")
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=config['random_state'])

        log("🔧 数据预处理...")
        preprocessor = DataPreprocessor(config)
        X_train = preprocessor.fit_transform(X_train_orig)
        X_test = preprocessor.transform(X_test_orig)
        result['processed_features'] = len(preprocessor.feature_names_out)
        result['preprocessing_info'] = (
            f"连续特征数: {len(preprocessor.continuous_features)}, "
            f"离散特征数: {len(preprocessor.discrete_features)}, "
            f"处理后总特征数: {len(preprocessor.feature_names_out)}"
        )

        if config.get('do_eda', False):
            log("📊 执行探索性数据分析...")
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            eda_res = perform_eda(df, y, X, numeric_cols, config['target_col'], task_dir)
            result.update(eda_res)

        all_models = {
            'XGBoost': xgb.XGBRegressor(random_state=config['random_state']),
            'LightGBM': lgb.LGBMRegressor(random_state=config['random_state'], verbose=-1),
            'Random Forest': RandomForestRegressor(random_state=config['random_state']),
            'Gradient Boosting': GradientBoostingRegressor(random_state=config['random_state']),
            'Extra Trees': ExtraTreesRegressor(random_state=config['random_state']),
            'AdaBoost': AdaBoostRegressor(random_state=config['random_state']),
            'Support Vector Machine': SVR(),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=config['random_state']),
            'Lasso Regression': Lasso(random_state=config['random_state']),
            'ElasticNet': ElasticNet(random_state=config['random_state']),
            'Decision Tree': DecisionTreeRegressor(random_state=config['random_state']),
            'K-Nearest Neighbors': KNeighborsRegressor()
        }
        selected = config.get('selected_models', list(all_models.keys()))
        models = {name: all_models[name] for name in selected if name in all_models}
        if not models:
            return {'status': 'error', 'message': '未选择任何模型'}

        cv_folds = config['cv_folds']
        model_results = {}
        trained_models = {}
        model_files = {}
        detailed_cv_scores = {}

        total_models = len(models)
        for idx, (name, model) in enumerate(models.items()):
            log(f"🤖 训练模型 ({idx+1}/{total_models}): {name}")
            try:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                model_results[name] = {
                    'Train_R2': train_r2,
                    'Test_R2': test_r2,
                    'Test_RMSE': test_rmse,
                    'Test_MAE': test_mae,
                    'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std()
                }
                model_results[name]['params'] = model.get_params()
                full_pipe = Pipeline([
                    ('preprocessor', preprocessor.preprocessor),
                    ('model', model)
                ])
                full_pipe.fit(X_train_orig, y_train)
                model_path = os.path.join(task_dir, f'model_{name}.pkl')
                joblib.dump(full_pipe, model_path)
                model_files[name] = model_path
                trained_models[name] = full_pipe
                detailed_cv_scores[name] = cv_scores
                log(f"  完成 {name}: Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
            except Exception as e:
                log(f"  {name} 训练失败: {str(e)}")
                result['message'] += f'{name} 训练失败: {str(e)}\n'

        if not model_results:
            result['status'] = 'error'
            result['message'] += '所有模型训练失败。'
            return result

        results_df = pd.DataFrame(model_results).T.sort_values('Test_R2', ascending=False)
        best_model_name = results_df.index[0]

        if config.get('use_optuna', False):
            log("🔍 贝叶斯超参数优化...")
            opt_res = perform_optuna_optimization(
                X_train, y_train, best_model_name, cv_folds, config['random_state'],
                n_trials=config['optuna_trials'],
                callback=lambda trial_num, score: log(f"Optuna 试验 {trial_num}: R²={score:.4f}", 0))

            opt_fig = opt_res.pop('_opt_fig', None)
            opt_res.pop('optimization_history', None)

            if opt_fig:
                result['optimization_history'] = save_fig_to_task(opt_fig, 'optimization_history.png')
            else:
                result['optimization_history'] = None

            result.update(opt_res)
            if opt_res.get('best_params') and opt_res['best_params'] != {}:
                best_params = opt_res['best_params']
                try:
                    optimized_model = build_optimized_model(best_model_name, best_params, config['random_state'])
                    optimized_model.fit(X_train, y_train)
                    full_pipe_opt = Pipeline([
                        ('preprocessor', preprocessor.preprocessor),
                        ('model', optimized_model)
                    ])
                    full_pipe_opt.fit(X_train_orig, y_train)
                    opt_model_path = os.path.join(task_dir, f'model_{best_model_name}_optimized.pkl')
                    joblib.dump(full_pipe_opt, opt_model_path)
                    model_files[best_model_name] = opt_model_path
                    trained_models[best_model_name] = full_pipe_opt
                    model_results[best_model_name]['params'] = best_params
                    result['best_model_params'] = json.dumps(best_params, indent=2, default=str)
                    log("✅ 优化模型已保存")
                except Exception as e:
                    log(f"优化模型创建失败，保留原模型: {str(e)}")
                    result['message'] += f'优化模型创建失败，保留原模型: {str(e)}\n'
                    result['best_model_params'] = '优化失败，使用默认参数'
            else:
                result['best_model_params'] = '该模型无可调超参数（已跳过优化）'
        else:
            result['best_model_params'] = '未启用优化（使用默认参数）'

        # 生成性能对比图
        log("📈 生成性能图表...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].barh(results_df.index, results_df['Test_R2'], color='skyblue')
        axes[0].set_xlabel('Test R²'); axes[0].set_title('Test R² by Model')
        axes[1].barh(results_df.index, results_df['Test_RMSE'], color='salmon')
        axes[1].set_xlabel('Test RMSE'); axes[1].set_title('Test RMSE by Model')
        axes[2].barh(results_df.index, results_df['CV_Mean'], xerr=results_df['CV_Std'], color='lightgreen')
        axes[2].set_xlabel(f'{cv_folds}-Fold CV R² Mean')
        axes[2].set_title('Cross Validation R²')
        plt.tight_layout()
        result['performance_chart'] = save_fig_to_task(fig, 'performance.png')

        # 统计显著性检验
        if config.get('do_stat_tests', False) and len(detailed_cv_scores) >= 2:
            log("📊 统计显著性检验...")
            stat_res = perform_statistical_tests(detailed_cv_scores, best_model_name, task_dir)
            result.update(stat_res)

        # SHAP 分析
        do_shap_adv = config.get('do_shap_advanced', False)
        best_pipe = trained_models[best_model_name]
        log("🧠 SHAP 分析...")
        shap_res = perform_shap_analysis(best_pipe, X_test, best_model_name, task_dir, do_advanced=do_shap_adv)
        result.update(shap_res)

        # 保存模型清单
        log("💾 保存模型清单...")
        for model_name, info in model_results.items():
            if 'params' in info:
                cleaned_params = {}
                for k, v in info['params'].items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        cleaned_params[k] = None
                    else:
                        cleaned_params[k] = v
                info['params'] = cleaned_params

        manifest = {
            'models': list(model_results.keys()),
            'metrics': model_results,
            'files': {k: os.path.basename(v) for k, v in model_files.items()},
            'feature_names': X_train_orig.columns.tolist(),
            'best_model': best_model_name
        }

        # 收集图表路径
        result_charts = {}
        for key, value in result.items():
            if isinstance(value, str) and value.startswith('/ml-tool/static/') and key not in ('manifest_file',):
                result_charts[key] = value
        if result.get('optimization_history'):
            result_charts['optimization_history'] = result['optimization_history']
        if result.get('cv_boxplot'):
            result_charts['cv_boxplot'] = result['cv_boxplot']
        for i in range(1, 4):
            key = f'shap_waterfall_{i}'
            if result.get(key):
                result_charts[key] = result[key]
        result_charts = {k: v for k, v in result_charts.items() if v}
        if result_charts:
            manifest['charts'] = result_charts

        # 统计文本
        stat_texts = {}
        for key in ('normality_test', 'ttest_results', 'wilcoxon_results'):
            if result.get(key):
                stat_texts[key] = result[key]
        if stat_texts:
            manifest['stat_texts'] = stat_texts

        manifest_path = os.path.join(task_dir, 'models_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str)
        result['manifest_file'] = manifest_path

        # 保存测试数据
        test_data_path = os.path.join(task_dir, 'test_data.pkl')
        joblib.dump((X_test_orig, y_test), test_data_path)

        result['models_list'] = results_df.index.tolist()
        result['best_model'] = best_model_name
        result['feature_names_original'] = X_train_orig.columns.tolist()
        result['prediction_example'] = X_test_orig.iloc[0].to_dict()

        # 保存填充值
        try:
            fill_values = {}
            for name, trans, cols in preprocessor.preprocessor.transformers_:
                if name == 'cont' and isinstance(trans, Pipeline):
                    steps = dict(trans.steps)
                    if 'imputer' in steps:
                        stats = steps['imputer'].statistics_
                        for col, val in zip(cols, stats):
                            fill_values[col] = float(val) if not np.isnan(val) else 0.0
                elif name == 'disc' and isinstance(trans, Pipeline):
                    steps = dict(trans.steps)
                    if 'imputer' in steps:
                        stats = steps['imputer'].statistics_
                        for col, val in zip(cols, stats):
                            fill_values[col] = val
            fill_path = os.path.join(task_dir, 'fill_values.json')
            with open(fill_path, 'w') as f:
                json.dump(fill_values, f, default=str)
        except:
            pass

        log("✅ 全流程完成！")
        return result

    except Exception:
        result['status'] = 'error'
        result['message'] = traceback.format_exc()
        return result


# ================== 辅助函数实现 ==================

def build_optimized_model(model_name, best_params, random_state):
    if model_name == 'XGBoost':
        return xgb.XGBRegressor(**best_params)
    elif model_name == 'LightGBM':
        return lgb.LGBMRegressor(**best_params, verbose=-1)
    elif model_name == 'Random Forest':
        return RandomForestRegressor(**best_params)
    elif model_name == 'Gradient Boosting':
        return GradientBoostingRegressor(**best_params)
    elif model_name == 'Extra Trees':
        return ExtraTreesRegressor(**best_params)
    elif model_name == 'AdaBoost':
        return AdaBoostRegressor(**best_params)
    elif model_name == 'Support Vector Machine':
        return SVR(**best_params)
    elif model_name == 'Ridge Regression':
        return Ridge(**best_params)
    elif model_name == 'Lasso Regression':
        return Lasso(**best_params)
    elif model_name == 'ElasticNet':
        return ElasticNet(**best_params)
    elif model_name == 'Decision Tree':
        return DecisionTreeRegressor(**best_params)
    elif model_name == 'K-Nearest Neighbors':
        return KNeighborsRegressor(**best_params)
    else:
        raise ValueError(f'不支持的模型: {model_name}')


def perform_eda(df, y, X, numeric_cols, target_col, task_dir=''):
    res = {}
    def save_or_b64(fig, fname):
        if task_dir:
            path = os.path.join(task_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            task_id = os.path.basename(task_dir)
            return f'/ml-tool/static/{task_id}/{fname}'
        else:
            return fig_to_base64(fig)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    miss_df = pd.DataFrame({'缺失数': missing, '缺失率': missing_pct}).sort_values('缺失数', ascending=False)
    res['missing_table'] = miss_df[miss_df['缺失数'] > 0].to_html(classes='table table-sm')
    res['desc_stats'] = df[numeric_cols].describe().to_html(classes='table table-sm')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax1.axvline(y.mean(), color='red', linestyle='--', label=f'均值={y.mean():.3f}')
    ax1.axvline(y.median(), color='orange', linestyle='--', label=f'中位数={y.median():.3f}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title('目标变量分布直方图')
    y.plot.kde(ax=ax2, color='purple', linewidth=2)
    ax2.axvline(y.mean(), color='red', linestyle='--')
    ax2.axvline(y.median(), color='orange', linestyle='--')
    ax2.set_title('核密度估计')
    stats.probplot(y, dist="norm", plot=ax3)
    ax3.set_title('Q-Q 图')
    ax4.boxplot(y, patch_artist=True)
    ax4.set_title('箱线图')
    plt.suptitle(f'目标变量分析: {target_col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    res['target_dist_plot'] = save_or_b64(fig, 'target_dist.png')

    try:
        if len(y) <= 5000:
            shapiro_stat, shapiro_p = shapiro(y.sample(min(5000, len(y))))
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        dagostino_stat, dagostino_p = normaltest(y)
        jb_stat, jb_p = jarque_bera(y)
        res['normality_text'] = (
            f"Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4e}\n"
            f"D'Agostino: stat={dagostino_stat:.4f}, p={dagostino_p:.4e}\n"
            f"Jarque-Bera: stat={jb_stat:.4f}, p={jb_p:.4e}"
        )
    except:
        res['normality_text'] = '正态性检验失败'

    corr_matrix = pd.concat([X[numeric_cols], y], axis=1).corr()
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, xticklabels=True, yticklabels=True, ax=ax_corr)
    ax_corr.set_title('特征相关性热力图')
    res['corr_heatmap'] = save_or_b64(fig_corr, 'corr_heatmap.png')

    top6 = target_corr.head(6).index.tolist()
    if len(top6) > 0:
        fig_scatter, axes_scatter = plt.subplots(2, 3, figsize=(15, 10))
        axes_scatter = axes_scatter.flatten()
        for i, feat in enumerate(top6):
            axes_scatter[i].scatter(X[feat], y, alpha=0.6)
            axes_scatter[i].set_xlabel(feat)
            axes_scatter[i].set_ylabel(target_col)
            axes_scatter[i].set_title(f'{feat} vs {target_col}')
            axes_scatter[i].grid(True, alpha=0.3)
        plt.suptitle('Top 6 特征与目标变量散点图', fontsize=14, fontweight='bold')
        plt.tight_layout()
        res['top6_scatter'] = save_or_b64(fig_scatter, 'top6_scatter.png')

    n_feat = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n_feat / ncols))
    fig_dist, axes_dist = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes_dist = axes_dist.flatten() if n_feat > 1 else [axes_dist]
    for i, feat in enumerate(numeric_cols):
        X[feat].hist(bins=30, ax=axes_dist[i], alpha=0.7, color='lightblue', edgecolor='black')
        axes_dist[i].set_title(f'{feat}\n偏度={X[feat].skew():.2f}')
    for j in range(n_feat, len(axes_dist)):
        axes_dist[j].set_visible(False)
    plt.suptitle('数值特征分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    res['feature_dist'] = save_or_b64(fig_dist, 'feature_dist.png')

    fig_box, axes_box = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes_box = axes_box.flatten() if n_feat > 1 else [axes_box]
    for i, feat in enumerate(numeric_cols):
        axes_box[i].boxplot(X[feat])
        axes_box[i].set_title(feat)
    for j in range(n_feat, len(axes_box)):
        axes_box[j].set_visible(False)
    plt.suptitle('离群值检测 (IQR)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    res['outlier_boxplot'] = save_or_b64(fig_box, 'outlier_boxplot.png')

    return res


def perform_statistical_tests(detailed_cv_scores, best_model_name, task_dir=''):
    res = {}
    def save_or_b64(fig, fname):
        if task_dir:
            path = os.path.join(task_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            task_id = os.path.basename(task_dir)
            return f'/ml-tool/static/{task_id}/{fname}'
        else:
            return fig_to_base64(fig)

    top5 = sorted(detailed_cv_scores.items(), key=lambda x: x[1].mean(), reverse=True)[:5]
    top5_dict = dict(top5)
    best_scores = detailed_cv_scores[best_model_name]

    normality_text = "Shapiro-Wilk 正态性检验:\n"
    for name, scores in top5_dict.items():
        stat, p = stats.shapiro(scores)
        normality_text += f"{name}: stat={stat:.4f}, p={p:.4e} {'正常' if p > 0.05 else '非正态'}\n"
    res['normality_test'] = normality_text

    ttest_text = f"配对t检验 (基准: {best_model_name}):\n"
    for name in top5_dict:
        if name == best_model_name:
            continue
        t_stat, p_val = ttest_rel(best_scores, top5_dict[name])
        ttest_text += f"{best_model_name} vs {name}: t={t_stat:.4f}, p={p_val:.4e} {'显著' if p_val < 0.05 else '不显著'}\n"
    res['ttest_results'] = ttest_text

    wilcox_text = f"Wilcoxon符号秩检验 (基准: {best_model_name}):\n"
    for name in top5_dict:
        if name == best_model_name:
            continue
        w_stat, p_val = wilcoxon(best_scores, top5_dict[name])
        wilcox_text += f"{best_model_name} vs {name}: W={w_stat:.4f}, p={p_val:.4e} {'显著' if p_val < 0.05 else '不显著'}\n"
    res['wilcoxon_results'] = wilcox_text

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [top5_dict[m] for m in top5_dict.keys()]
    labels = list(top5_dict.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('CV R²')
    ax.set_title('交叉验证得分分布')
    for i, name in enumerate(labels[1:], 1):
        _, p = ttest_rel(best_scores, data[i])
        y_pos = max(data[i]) * 1.02
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        ax.text(i + 1, y_pos, sig, ha='center', fontweight='bold')
    plt.tight_layout()
    res['cv_boxplot'] = save_or_b64(fig, 'cv_boxplot.png')
    return res


def perform_optuna_optimization(X_train, y_train, model_name, cv_folds, random_state, n_trials=100, callback=None):
    # 移除 no_tune 限制，所有模型均可参与优化
    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state
            }
            model = xgb.XGBRegressor(**params)
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state,
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = RandomForestRegressor(**params)
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': random_state
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == 'Extra Trees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = ExtraTreesRegressor(**params)
        elif model_name == 'AdaBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'random_state': random_state
            }
            model = AdaBoostRegressor(**params)
        elif model_name == 'Support Vector Machine':
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
            }
            model = SVR(**params)
        elif model_name == 'Ridge Regression':
            params = {'alpha': trial.suggest_float('alpha', 0.01, 10, log=True)}
            model = Ridge(**params, random_state=random_state)
        elif model_name == 'Lasso Regression':
            params = {'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True)}
            model = Lasso(**params, random_state=random_state)
        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
            }
            model = ElasticNet(**params, random_state=random_state)
        elif model_name == 'Decision Tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = DecisionTreeRegressor(**params)
        elif model_name == 'K-Nearest Neighbors':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 2, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
            }
            model = KNeighborsRegressor(**params)
        else:
            return 0.0

        cv_score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2').mean()
        if callback:
            callback(trial.number, cv_score)
        return cv_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig_opt = ax.figure if hasattr(ax, 'figure') else ax
    res = {
        'best_params': study.best_params,
        'best_cv_score': study.best_value,
        'optimization_history': None
    }
    res['_opt_fig'] = fig_opt
    return res


def perform_shap_analysis(pipeline, X_test, model_name, task_dir='', do_advanced=False):
    res = {}
    def save_or_b64(fig, fname):
        if task_dir:
            path = os.path.join(task_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            task_id = os.path.basename(task_dir)
            return f'/ml-tool/static/{task_id}/{fname}'
        else:
            return fig_to_base64(fig)

    try:
        model = pipeline.named_steps['model']
        if model_name in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting',
                          'Extra Trees', 'Decision Tree']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            expected_value = explainer.expected_value
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, 50))
            shap_values = explainer.shap_values(X_test, nsamples=100)
            expected_value = explainer.expected_value

        fig_summary = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        res['shap_summary'] = save_or_b64(fig_summary, 'shap_summary.png')

        if do_advanced:
            n_samples = min(50, len(X_test))
            n_features = min(10, X_test.shape[1])
            shap_importance = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(shap_importance)[::-1]
            top_features = X_test.columns[sorted_idx[:n_features]]
            fig_dec = plt.figure()
            shap.decision_plot(expected_value, shap_values[:n_samples, sorted_idx[:n_features]],
                               X_test.iloc[:n_samples, sorted_idx[:n_features]],
                               feature_names=top_features, show=False)
            res['shap_decision'] = save_or_b64(fig_dec, 'shap_decision.png')

            for idx in range(min(3, len(X_test))):
                fig_wf = plt.figure()
                shap.plots.waterfall(
                    shap.Explanation(values=shap_values[idx],
                                     base_values=expected_value,
                                     data=X_test.iloc[idx],
                                     feature_names=X_test.columns),
                    show=False)
                res[f'shap_waterfall_{idx+1}'] = save_or_b64(fig_wf, f'shap_waterfall_{idx+1}.png')

            fig_heat = plt.figure()
            shap.plots.heatmap(
                shap.Explanation(values=shap_values[:min(20, len(X_test))],
                                 data=X_test.iloc[:min(20, len(X_test))].values,
                                 feature_names=X_test.columns),
                show=False)
            res['shap_heatmap'] = save_or_b64(fig_heat, 'shap_heatmap.png')

            shap_imp_mean = np.abs(shap_values).mean(axis=0)
            fig_clust = plt.figure(figsize=(10, 6))
            linkage_matrix = linkage(shap_imp_mean.reshape(-1, 1), method='ward')
            dendrogram(linkage_matrix, labels=X_test.columns.tolist(), orientation='left')
            plt.title('基于SHAP重要性的特征聚类')
            plt.tight_layout()
            res['shap_clustering'] = save_or_b64(fig_clust, 'shap_clustering.png')

            top6_idx = sorted_idx[:6]
            fig_dist, axes_dist = plt.subplots(2, 3, figsize=(15, 10))
            axes_dist = axes_dist.flatten()
            for i, fidx in enumerate(top6_idx):
                axes_dist[i].hist(shap_values[:, fidx], bins=30, alpha=0.7)
                axes_dist[i].axvline(0, color='black', linestyle='-')
                axes_dist[i].set_title(X_test.columns[fidx])
            plt.suptitle('SHAP值分布 (Top6)')
            plt.tight_layout()
            res['shap_distributions'] = save_or_b64(fig_dist, 'shap_distributions.png')

            shap_corr = np.corrcoef(shap_values.T)
            fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(shap_corr, dtype=bool))
            sns.heatmap(shap_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                        xticklabels=X_test.columns, yticklabels=X_test.columns, ax=ax_corr)
            ax_corr.set_title('SHAP值相关性矩阵')
            plt.tight_layout()
            res['shap_corr_matrix'] = save_or_b64(fig_corr, 'shap_corr_matrix.png')

            n_bootstrap = 20
            bootstrap_means = []
            for _ in range(n_bootstrap):
                samp_idx = np.random.choice(len(X_test), size=int(0.8 * len(X_test)), replace=True)
                bootstrap_means.append(np.abs(shap_values[samp_idx]).mean(axis=0))
            bootstrap_means = np.array(bootstrap_means)
            mean_imp = bootstrap_means.mean(axis=0)
            std_imp = bootstrap_means.std(axis=0)
            fig_stab, ax_stab = plt.subplots(figsize=(12, 6))
            ax_stab.errorbar(range(len(X_test.columns)), mean_imp, yerr=std_imp, fmt='o')
            ax_stab.set_xticks(range(len(X_test.columns)))
            ax_stab.set_xticklabels(X_test.columns, rotation=45, ha='right')
            ax_stab.set_ylabel('Mean |SHAP|')
            ax_stab.set_title('SHAP稳定性 (Bootstrap)')
            plt.tight_layout()
            res['shap_stability'] = save_or_b64(fig_stab, 'shap_stability.png')

    except Exception as e:
        res['shap_error'] = f'SHAP分析异常: {str(e)}'
    return res


def generate_model_chart(model_name, chart_type, manifest_path='uploads/models_manifest.json'):
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        task_dir = os.path.dirname(manifest_path)
        model_file = os.path.join(task_dir, manifest['files'][model_name])
        if not os.path.exists(model_file):
            return None
        pipe = joblib.load(model_file)
        test_data_path = os.path.join(task_dir, 'test_data.pkl')
        if not os.path.exists(test_data_path):
            return None
        X_test_orig, y_test = joblib.load(test_data_path)
        y_pred = pipe.predict(X_test_orig)

        if chart_type == 'fit':
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(f'{model_name} 拟合图')
            ax.grid(True)
            return fig
        elif chart_type == 'importance':
            model = pipe.named_steps['model']
            preprocessor = pipe.named_steps['preprocessor']
            try:
                feature_names_out = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names_out = manifest['feature_names']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(importances) != len(feature_names_out):
                    feature_names_out = manifest['feature_names'][:len(importances)]
                feat_imp = pd.Series(importances, index=feature_names_out).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 6))
                feat_imp.plot.barh(ax=ax, color='teal')
                ax.invert_yaxis()
                ax.set_title(f'{model_name} 特征重要性 (Top 15)')
                return fig
    except Exception as e:
        print(f"生成图表失败: {e}")
    return None