# seal_monitor/routes.py
import os
import json
import uuid
import shutil
import sqlite3
from datetime import datetime
from flask import request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
from config import Config
from .utils import get_seal_db, init_seal_db, PARAM_LABELS
from .tasks import start_analysis, get_task_status, cleanup_old_tasks
from . import seal_bp
import secrets


def ensure_db_init():
    """确保数据库已初始化（懒加载）"""
    try:
        init_seal_db()
    except Exception as e:
        print(f"数据库初始化警告: {e}")


def get_devices():
    ensure_db_init()
    db = get_seal_db()
    try:
        rows = db.execute('SELECT device_id, device_name FROM devices ORDER BY device_id').fetchall()
        return [{'device_id': r['device_id'], 'device_name': r['device_name']} for r in rows]
    finally:
        pass


@seal_bp.route('/')
def seal_dashboard():
    ensure_db_init()
    return render_template('seal_dashboard.html')


@seal_bp.route('/api/devices/list')
def api_devices_list():
    ensure_db_init()
    return jsonify(get_devices())


@seal_bp.route('/api/devices/add', methods=['POST'])
def api_devices_add():
    ensure_db_init()
    data = request.get_json()
    device_name = data.get('device_name', '').strip()
    if not device_name:
        return jsonify({'error': '设备名称不能为空'}), 400

    device_id = 'dev_' + datetime.now().strftime('%Y%m%d%H%M%S') + '_' + str(uuid.uuid4())[:4]

    db = get_seal_db()
    try:
        db.execute('INSERT INTO devices (device_id, device_name) VALUES (?, ?)',
                   (device_id, device_name))
        db.commit()
        os.makedirs(os.path.join(Config.SEAL_DATA_DIR, device_id), exist_ok=True)
        os.makedirs(os.path.join(Config.SEAL_MODEL_DIR, device_id), exist_ok=True)
        return jsonify({'status': 'ok', 'device_id': device_id, 'device_name': device_name})
    except sqlite3.IntegrityError:
        return jsonify({'error': '设备ID冲突'}), 500


@seal_bp.route('/api/devices/delete/<device_id>', methods=['DELETE'])
def api_devices_delete(device_id):
    ensure_db_init()
    db = get_seal_db()
    db.execute('DELETE FROM devices WHERE device_id=?', (device_id,))
    db.execute('DELETE FROM analysis_history WHERE device_id=?', (device_id,))
    db.execute('DELETE FROM device_config WHERE device_id=?', (device_id,))
    db.commit()

    data_dir = os.path.join(Config.SEAL_DATA_DIR, device_id)
    model_dir = os.path.join(Config.SEAL_MODEL_DIR, device_id)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    return jsonify({'status': 'deleted'})


@seal_bp.route('/api/param_labels')
def api_param_labels():
    return jsonify(PARAM_LABELS)


@seal_bp.route('/api/config/<device_id>', methods=['GET', 'POST'])
def device_config(device_id):
    ensure_db_init()
    db = get_seal_db()
    if request.method == 'GET':
        row = db.execute('SELECT config_json FROM device_config WHERE device_id=?', (device_id,)).fetchone()
        default = {}
        for key in dir(Config):
            if key.startswith('SEAL_') and isinstance(getattr(Config, key), (int, float, str, bool, list, dict)):
                default[key] = getattr(Config, key)
        if row:
            saved = json.loads(row['config_json'])
            default.update(saved)
        return jsonify(default)
    else:
        config = request.get_json()
        if not isinstance(config, dict):
            return jsonify({"error": "Invalid JSON"}), 400
        typed_config = {}
        for key, value in config.items():
            default_type = type(getattr(Config, key, None))
            if default_type in (int, float, bool, list, dict):
                try:
                    if default_type == int:
                        typed_config[key] = int(value)
                    elif default_type == float:
                        typed_config[key] = float(value)
                    elif default_type == bool:
                        typed_config[key] = bool(value)
                    else:
                        typed_config[key] = value
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for {key}"}), 400
            else:
                typed_config[key] = value
        db.execute('INSERT OR REPLACE INTO device_config (device_id, config_json) VALUES (?, ?)',
                   (device_id, json.dumps(typed_config)))
        db.commit()
        return jsonify({'status': 'saved'})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.SEAL_ALLOWED_EXTENSIONS


def validate_csv(file_stream):
    try:
        import csv as csv_module
        import io
        sample = file_stream.read(2048)
        file_stream.seek(0)
        if not sample.strip():
            return False
        csv_module.Sniffer().sniff(sample.decode('utf-8', errors='ignore')[:1024])
        lines = sample.decode('utf-8').splitlines()
        if len(lines) < 2:
            return False
        reader = csv_module.reader(io.StringIO(sample.decode('utf-8')))
        header = [col.lower().strip() for col in next(reader)]
        if 'timestamp' not in header:
            return False
        return True
    except:
        return False


@seal_bp.route('/upload', methods=['POST'])
def upload():
    ensure_db_init()
    device_id = request.form['device_id']
    devices = [d['device_id'] for d in get_devices()]
    if device_id not in devices:
        return jsonify({"error": "Invalid device"}), 400
    files = request.files.getlist('files')
    saved = 0
    errors = []
    for file in files:
        if file and allowed_file(file.filename) and validate_csv(file):
            safe_name = secure_filename(file.filename) or 'upload.csv'
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            unique_name = f"{timestamp}_{safe_name}"
            save_path = os.path.join(Config.SEAL_DATA_DIR, device_id, unique_name)
            file.save(save_path)
            saved += 1
        else:
            errors.append(f"{file.filename} 无效")
    return jsonify({"uploaded": saved, "errors": errors})


@seal_bp.route('/api/analyze_async/<device_id>', methods=['POST'])
def analyze_async(device_id):
    ensure_db_init()
    devices = [d['device_id'] for d in get_devices()]
    if device_id not in devices:
        return jsonify({"error": "Invalid device"}), 400
    config_overrides = request.get_json() or {}
    task_id = start_analysis(device_id, config_overrides)
    return jsonify({'task_id': task_id})


@seal_bp.route('/api/task/<task_id>')
def task_status(task_id):
    task = get_task_status(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({
        'status': task['status'],
        'result': task['result'],
        'db_id': task.get('db_id'),
        'progress': task.get('progress')
    })


@seal_bp.route('/api/history/<device_id>')
def api_history(device_id):
    ensure_db_init()
    db = get_seal_db()
    rows = db.execute('SELECT * FROM analysis_history WHERE device_id=? ORDER BY analysis_time DESC LIMIT 20',
                      (device_id,)).fetchall()
    hist = [{
        'id': r['id'],
        'time': r['analysis_time'],
        'avg_health': r['avg_health'],
        'rul': r['rul'],
        'anomaly_count': r['anomaly_count']
    } for r in rows]
    return jsonify(hist)


@seal_bp.route('/api/report/<int:analysis_id>')
def api_report(analysis_id):
    ensure_db_init()
    db = get_seal_db()
    row = db.execute('SELECT * FROM analysis_history WHERE id=?', (analysis_id,)).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    result = json.loads(row['result_json'])
    result['id'] = analysis_id
    return jsonify(result)


@seal_bp.route('/api/delete_record/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    ensure_db_init()
    db = get_seal_db()
    db.execute('DELETE FROM analysis_history WHERE id=?', (record_id,))
    db.commit()
    return jsonify({'status': 'deleted'})


@seal_bp.route('/download/report/<int:analysis_id>')
def download_report(analysis_id):
    ensure_db_init()
    db = get_seal_db()
    row = db.execute('SELECT result_json FROM analysis_history WHERE id=?', (analysis_id,)).fetchone()
    if not row:
        return "Not found", 404
    result = json.loads(row['result_json'])
    import io
    import csv as csv_module
    si = io.StringIO()
    writer = csv_module.writer(si)
    writer.writerow(['Date', 'Health Score', 'Anomaly Score'])
    for date, h, s in zip(result['dates'], result['health_scores'], result['vae_scores']):
        writer.writerow([date, h, s])
    output = si.getvalue()
    return send_file(
        io.BytesIO(output.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"{result['device_id']}_report.csv"
    )


@seal_bp.route('/api/devices/status')
def devices_status():
    ensure_db_init()
    db = get_seal_db()
    all_devices = get_devices()
    status = {}
    for dev in all_devices:
        device_id = dev['device_id']
        row = db.execute('SELECT * FROM analysis_history WHERE device_id=? ORDER BY analysis_time DESC LIMIT 1',
                         (device_id,)).fetchone()
        if row:
            res = json.loads(row['result_json'])
            avg_health = res.get('avg_health')
            status[device_id] = {
                'device_name': dev['device_name'],
                'avg_health': avg_health,
                'rul': res.get('rul'),
                'anomaly_count': res.get('anomaly_count', 0),
                'status': 'Normal' if avg_health and avg_health > Config.SEAL_HEALTH_NORMAL
                else 'Warning' if avg_health and avg_health > Config.SEAL_HEALTH_WARNING
                else 'Alarm'
            }
        else:
            status[device_id] = {
                'device_name': dev['device_name'],
                'avg_health': None,
                'rul': None,
                'anomaly_count': 0,
                'status': 'Unknown'
            }
    return jsonify(status)


# 清理过期任务
@seal_bp.before_request
def cleanup():
    cleanup_old_tasks()