import os
import io
import json
import uuid
import threading
import time
import shutil
import re
import math
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, Response, send_file, send_from_directory, current_app
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from modeling import run_full_pipeline, generate_model_chart

ml_tool_bp = Blueprint('ml_tool', __name__, url_prefix='/ml-tool')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_TRAIN_ROWS = 200000          # CSV 最大行数限制
MAX_CONCURRENT_TASKS = 2        # 最大并行训练任务数

training_tasks = {}
task_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)


def validate_task_id(task_id):
    """只允许 UUID 格式或 32 位十六进制字符串，防止路径遍历"""
    if re.fullmatch(r'[0-9a-f]{32}', task_id):
        return True
    try:
        uuid.UUID(task_id)
        return True
    except (ValueError, AttributeError):
        return False


def update_task(task_id, **kwargs):
    with task_lock:
        if task_id in training_tasks:
            task = training_tasks[task_id]
            task.update(kwargs)
            if task['status'] in ('completed', 'failed') and 'finished_at' not in task:
                task['finished_at'] = time.time()


def clean_old_tasks():
    now = time.time()
    to_delete = []
    with task_lock:
        for tid, task in list(training_tasks.items()):
            if task['status'] in ('pending', 'running'):
                if now - task.get('created_at', now) > 3600:
                    to_delete.append(tid)
                continue
            finished = task.get('finished_at')
            if finished and now - finished > 1800:
                to_delete.append(tid)
            elif not finished:
                to_delete.append(tid)
        for tid in to_delete:
            if tid in training_tasks:
                del training_tasks[tid]


def run_training_task(task_id, config, app):
    task_dir = os.path.join(UPLOAD_FOLDER, task_id)
    os.makedirs(task_dir, exist_ok=True)
    data_path = config['data_path']
    upload_real = os.path.realpath(UPLOAD_FOLDER)
    if not os.path.realpath(data_path).startswith(upload_real):
        update_task(task_id, status='failed', result={'message': '非法的数据文件路径'},
                    logs=[f"[{time.strftime('%H:%M:%S')}] ❌ 路径不合法"], finished_at=time.time())
        return
    if os.path.dirname(data_path) != task_dir:
        new_data_path = os.path.join(task_dir, os.path.basename(data_path))
        shutil.copy2(data_path, new_data_path)
        config['data_path'] = new_data_path
    config['task_dir'] = task_dir

    with app.app_context():
        try:
            update_task(task_id, status='running', progress=0, logs=[])
            logs = []

            def progress_callback(msg, progress_increment=1):
                logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                with task_lock:
                    task = training_tasks.get(task_id)
                    if not task:
                        return
                    new_progress = min(task.get('progress', 0) + progress_increment, 95)
                    task['logs'] = logs
                    task['progress'] = new_progress

            result = run_full_pipeline(config, progress_callback=progress_callback)

            if result['status'] == 'success':
                task_name = config.get('task_name', task_id[:8])
                manifest_path = os.path.join(task_dir, 'models_manifest.json')
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    manifest['task_name'] = task_name
                    with open(manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(manifest, f, indent=2, default=str)
                update_task(task_id, result=result, status='completed', progress=100,
                            logs=logs + [f"[{time.strftime('%H:%M:%S')}] ✅ 训练完成！"],
                            finished_at=time.time())
            else:
                raise Exception(result.get('message', '训练失败'))
        except Exception as e:
            logs.append(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: {str(e)}")
            update_task(task_id, status='failed', result={'message': str(e)},
                        logs=logs, finished_at=time.time())


@ml_tool_bp.route('/')
def index():
    return render_template('ml_tool.html')


@ml_tool_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '未找到文件'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '文件名为空'})
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': '仅支持CSV文件'})
    filename = secure_filename(file.filename)
    ext = '.csv'
    saved_hex = uuid.uuid4().hex
    saved_name = saved_hex + ext
    filepath = os.path.join(UPLOAD_FOLDER, saved_name)
    file.save(filepath)
    try:
        df = pd.read_csv(filepath)
        if len(df) > MAX_TRAIN_ROWS:
            os.unlink(filepath)
            return jsonify({'status': 'error', 'message': f'文件行数不能超过 {MAX_TRAIN_ROWS}，当前 {len(df)} 行'})
        columns = df.columns.tolist()
        candidates = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                n_unique = df[col].nunique()
                if n_unique <= 15:
                    candidates[col] = int(n_unique)
            elif pd.api.types.is_string_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
                n_unique = df[col].nunique()
                if n_unique <= 15:
                    candidates[col] = int(n_unique)
        return jsonify({
            'status': 'success',
            'file_token': saved_hex,  # 只返回 hex 部分
            'columns': columns,
            'discrete_candidates': candidates
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'CSV解析失败: {str(e)}'})


@ml_tool_bp.route('/train', methods=['POST'])
def train_async():
    # 解析文件令牌，构建绝对路径
    file_token = request.form.get('file_token', '')
    if not file_token:
        return jsonify({'error': '缺少文件令牌'}), 400
    if not re.fullmatch(r'[0-9a-f]{32}', file_token):
        return jsonify({'error': '无效的文件令牌'}), 400
    data_path = os.path.join(UPLOAD_FOLDER, file_token + '.csv')  # 加上 .csv
    if not os.path.exists(data_path):
        return jsonify({'error': '文件不存在或已过期'}), 400
    if not os.path.realpath(data_path).startswith(os.path.realpath(UPLOAD_FOLDER)):
        return jsonify({'error': '非法文件路径'}), 400

    config = {
        'data_path': data_path,
        'target_col': request.form.get('target_col'),
        'drop_cols': request.form.get('drop_cols', ''),
        'test_size': float(request.form.get('test_size', 0.3)),
        'random_state': int(request.form.get('random_state', 42)),
        'cv_folds': int(request.form.get('cv_folds', 5)),
        'selected_models': request.form.getlist('selected_models'),
        'use_optuna': request.form.get('use_optuna') == 'true',
        'optuna_trials': int(request.form.get('optuna_trials', 50)),
        'do_eda': request.form.get('do_eda') == 'true',
        'do_stat_tests': request.form.get('do_stat_tests') == 'true',
        'do_shap_advanced': request.form.get('do_shap_advanced') == 'true',
        'missing_strategy': request.form.get('missing_strategy', 'median'),
        'missing_fill_value': float(request.form.get('missing_fill_value', 0)),
        'outlier_method': request.form.get('outlier_method', 'none'),
        'scaler': request.form.get('scaler', 'standard'),
        'discrete_columns': request.form.getlist('discrete_columns'),
        'auto_detect_discrete': request.form.get('auto_detect_discrete') == 'true',
        'discrete_encode': request.form.get('discrete_encode', 'onehot'),
        'discrete_max_cardinality': int(request.form.get('discrete_max_cardinality', 15)),
        'task_name': request.form.get('task_name', '').strip()
    }
    if not config['target_col']:
        return jsonify({'error': '请选择目标列'}), 400
    if not config['selected_models']:
        return jsonify({'error': '请至少选择一个模型'}), 400
    drop_cols_list = [c.strip() for c in config['drop_cols'].split(',') if c.strip()]
    if config['target_col'] in drop_cols_list:
        return jsonify({'error': f'目标列 "{config["target_col"]}" 不能出现在删除列中'}), 400

    # 并发控制
    with task_lock:
        running_count = sum(1 for t in training_tasks.values() if t['status'] == 'running')
        if running_count >= MAX_CONCURRENT_TASKS:
            return jsonify({'error': f'系统繁忙，当前并行训练任务已达上限（{MAX_CONCURRENT_TASKS}），请稍后再试'}), 503

    task_id = str(uuid.uuid4())
    with task_lock:
        training_tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'logs': [],
            'result': None,
            'created_at': time.time(),
            'task_name': config['task_name'] if config['task_name'] else task_id[:8]
        }
    app = current_app._get_current_object()
    executor.submit(run_training_task, task_id, config, app)   # 使用线程池
    return jsonify({'status': 'started', 'task_id': task_id})


@ml_tool_bp.route('/training_progress/<task_id>')
def training_progress(task_id):
    if not validate_task_id(task_id):
        return jsonify({'error': '非法任务ID'}), 400
    clean_old_tasks()
    with task_lock:
        task = training_tasks.get(task_id)
        if not task:
            return jsonify({'error': '任务不存在'}), 404
        resp = {
            'status': task['status'],
            'progress': task['progress'],
            'logs': task['logs'][:],
            'result': task.get('result')
        }
    return jsonify(resp)


@ml_tool_bp.route('/task_list')
def task_list():
    tasks_info = []
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify(tasks_info)
    for d in os.listdir(UPLOAD_FOLDER):
        dpath = os.path.join(UPLOAD_FOLDER, d)
        if os.path.isdir(dpath):
            manifest_path = os.path.join(dpath, 'models_manifest.json')
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    mtime = os.path.getmtime(manifest_path)
                    tasks_info.append({
                        'task_id': d,
                        'task_name': manifest.get('task_name', d[:8]),
                        'created_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)),
                        'num_models': len(manifest.get('models', [])),
                        'best_model': manifest.get('best_model', '未知')
                    })
                except:
                    continue
    tasks_info.sort(key=lambda x: x['created_time'], reverse=True)
    return jsonify(tasks_info)


@ml_tool_bp.route('/task/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    if not validate_task_id(task_id):
        return jsonify({'status': 'error', 'message': '非法任务ID'}), 400
    task_dir = os.path.join(UPLOAD_FOLDER, task_id)
    if not os.path.exists(task_dir):
        return jsonify({'status': 'error', 'message': '任务不存在'}), 404

    # 禁止删除正在运行的任务
    with task_lock:
        task = training_tasks.get(task_id)
        if task and task.get('status') == 'running':
            return jsonify({'status': 'error', 'message': '任务正在运行，无法删除'}), 400
        if task:
            del training_tasks[task_id]

    shutil.rmtree(task_dir, ignore_errors=True)
    return jsonify({'status': 'success', 'message': '任务已删除'})


@ml_tool_bp.route('/models_info')
def models_info():
    task_id = request.args.get('task_id')
    if not task_id:
        with task_lock:
            completed = [(tid, t) for tid, t in training_tasks.items() if t['status'] == 'completed']
            if completed:
                completed.sort(key=lambda x: x[1].get('finished_at', 0), reverse=True)
                task_id = completed[0][0]
            else:
                subdirs = []
                for d in os.listdir(UPLOAD_FOLDER):
                    dpath = os.path.join(UPLOAD_FOLDER, d)
                    if os.path.isdir(dpath) and os.path.exists(os.path.join(dpath, 'models_manifest.json')):
                        mtime = os.path.getmtime(os.path.join(dpath, 'models_manifest.json'))
                        subdirs.append((d, mtime))
                if subdirs:
                    subdirs.sort(key=lambda x: x[1], reverse=True)
                    task_id = subdirs[0][0]
                else:
                    return jsonify({'status': 'error', 'message': '暂无训练好的模型'})
    if not validate_task_id(task_id):
        return jsonify({'status': 'error', 'message': '非法任务ID'}), 400
    manifest_path = os.path.join(UPLOAD_FOLDER, task_id, 'models_manifest.json')
    if not os.path.exists(manifest_path):
        return jsonify({'status': 'error', 'message': '清单文件不存在'})
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    for model_name, info in manifest.get('metrics', {}).items():
        if 'params' in info:
            for k, v in info['params'].items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    info['params'][k] = None

    manifest['task_id'] = task_id
    manifest['static_base'] = f'/ml-tool/static/{task_id}'
    manifest['task_name'] = manifest.get('task_name', task_id[:8])
    # 新增字段，告知前端是否支持随机样本
    manifest['has_original_data'] = os.path.exists(os.path.join(UPLOAD_FOLDER, task_id, 'original_data.pkl'))
    return jsonify(manifest)


@ml_tool_bp.route('/static/<task_id>/<path:filename>')
def serve_static(task_id, filename):
    if not validate_task_id(task_id):
        return 'Invalid task_id', 400
    directory = os.path.join(UPLOAD_FOLDER, task_id)
    if not os.path.realpath(directory).startswith(os.path.realpath(UPLOAD_FOLDER)):
        return 'Forbidden', 403
    return send_from_directory(directory, filename)


@ml_tool_bp.route('/model_chart/<model_name>/<chart_type>')
def model_chart(model_name, chart_type):
    task_id = request.args.get('task_id')
    if not task_id or not validate_task_id(task_id):
        return '', 404
    manifest_path = os.path.join(UPLOAD_FOLDER, task_id, 'models_manifest.json')
    if not os.path.exists(manifest_path):
        return '', 404
    fig = generate_model_chart(model_name, chart_type, manifest_path)
    if fig is None:
        return '', 404
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png')


@ml_tool_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    task_id = data.get('task_id')
    if not task_id or not validate_task_id(task_id):
        return jsonify({'status': 'error', 'message': '缺少 task_id'})
    manifest_path = os.path.join(UPLOAD_FOLDER, task_id, 'models_manifest.json')
    if not os.path.exists(manifest_path):
        return jsonify({'status': 'error', 'message': '请先训练模型'})
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    model_name = data.get('model_name') or manifest.get('best_model', manifest['models'][0])
    if model_name not in manifest['files']:
        return jsonify({'status': 'error', 'message': f'模型 {model_name} 不存在'})
    model_path = os.path.join(UPLOAD_FOLDER, task_id, os.path.basename(manifest['files'][model_name]))
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': '模型文件丢失'})
    pipe = joblib.load(model_path)
    fill_path = os.path.join(UPLOAD_FOLDER, task_id, 'fill_values.json')
    fill_values = {}
    if os.path.exists(fill_path):
        with open(fill_path, 'r') as f:
            fill_values = json.load(f)
    input_dict = {}
    for feat in manifest['feature_names']:
        val = data.get(feat, '')
        if val is None or str(val).strip() == '':
            default = fill_values.get(feat, 0)
            try:
                input_dict[feat] = float(default)
            except:
                input_dict[feat] = 0.0
        else:
            try:
                fval = float(val)
                if not math.isfinite(fval):
                    raise ValueError('不是有限数')
                input_dict[feat] = fval
            except (ValueError, TypeError) as e:
                return jsonify({'status': 'error', 'message': f'特征 {feat} 的值非法: {val}'})
    input_df = pd.DataFrame([input_dict])
    try:
        pred = pipe.predict(input_df)
        return jsonify({'status': 'success', 'prediction': float(pred[0])})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@ml_tool_bp.route('/random_sample')
def random_sample():
    """从原始数据中随机抽取一条记录，用于预测界面一键填入"""
    task_id = request.args.get('task_id')
    if not task_id or not validate_task_id(task_id):
        return jsonify({'status': 'error', 'message': '缺少或无效 task_id'}), 400
    original_path = os.path.join(UPLOAD_FOLDER, task_id, 'original_data.pkl')
    if not os.path.exists(original_path):
        return jsonify({'status': 'error', 'message': '该任务未保存原始数据，不支持随机样本'}), 404
    try:
        df = pd.read_pickle(original_path)
        # 只保留特征列（训练时使用的列）
        manifest_path = os.path.join(UPLOAD_FOLDER, task_id, 'models_manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        features = manifest.get('feature_names', df.columns.tolist())
        available = [c for c in features if c in df.columns]
        if not available:
            return jsonify({'status': 'error', 'message': '无可用特征列'}), 500
        row = df[available].sample(1).iloc[0].to_dict()
        # 将numpy类型转换为Python原生类型（JSON序列化要求）
        row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
        return jsonify({'status': 'success', 'sample': row})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'随机抽样失败: {str(e)}'}), 500


@ml_tool_bp.route('/download_model/<model_name>')
def download_model(model_name):
    task_id = request.args.get('task_id')
    if not task_id or not validate_task_id(task_id):
        return '缺少任务ID', 400
    manifest_path = os.path.join(UPLOAD_FOLDER, task_id, 'models_manifest.json')
    if not os.path.exists(manifest_path):
        return '无模型', 404
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    if model_name in manifest.get('files', {}):
        file_path = os.path.join(UPLOAD_FOLDER, task_id, os.path.basename(manifest['files'][model_name]))
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=f'{model_name}.pkl')
    return '模型不存在', 404