import threading
import uuid
import time
import sqlite3
import json
from datetime import datetime
from .core import train_and_predict
from config import Config

tasks = {}
task_lock = threading.Lock()
MAX_CONCURRENT_TASKS = Config.SEAL_MAX_CONCURRENT_TASKS
semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_TASKS)

device_locks = {}
device_locks_lock = threading.Lock()

def get_device_lock(device_id):
    with device_locks_lock:
        if device_id not in device_locks:
            device_locks[device_id] = threading.Lock()
        return device_locks[device_id]

def save_result_to_db(device_id, result):
    conn = sqlite3.connect(Config.SEAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_json TEXT,
            avg_health REAL,
            rul TEXT,
            anomaly_count INTEGER
        )
    """)
    avg_health = result.get('avg_health')
    rul = result.get('rul')
    anomaly_count = result.get('anomaly_count', 0)
    cursor.execute(
        "INSERT INTO analysis_history (device_id, result_json, avg_health, rul, anomaly_count) VALUES (?, ?, ?, ?, ?)",
        (device_id, json.dumps(result), avg_health, str(rul) if rul else None, anomaly_count)
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def worker(task_id, device_id, config_overrides):
    acquired = semaphore.acquire(blocking=False)
    if not acquired:
        with task_lock:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['result'] = {'success': False, 'message': 'Server busy, try later'}
        return

    dev_lock = get_device_lock(device_id)
    if not dev_lock.acquire(blocking=False):
        semaphore.release()
        if not dev_lock.acquire(timeout=300):
            with task_lock:
                tasks[task_id]['status'] = 'failed'
                tasks[task_id]['result'] = {'success': False, 'message': 'Device busy, timeout'}
            return
        if not semaphore.acquire(blocking=False):
            dev_lock.release()
            with task_lock:
                tasks[task_id]['status'] = 'failed'
                tasks[task_id]['result'] = {'success': False, 'message': 'Server busy'}
            return

    def update_progress(current, total, message):
        with task_lock:
            if task_id in tasks:
                tasks[task_id]['progress'] = {
                    'current': current,
                    'total': total,
                    'message': message
                }

    try:
        with task_lock:
            tasks[task_id]['status'] = 'running'
            tasks[task_id]['progress'] = {'current': 0, 'total': 7, 'message': '开始分析'}

        result = train_and_predict(device_id, config_overrides, progress_callback=update_progress)

        db_id = None
        if result.get('success'):
            db_id = save_result_to_db(device_id, result)
        with task_lock:
            tasks[task_id]['status'] = 'completed' if result.get('success') else 'failed'
            tasks[task_id]['result'] = result
            tasks[task_id]['db_id'] = db_id
    except Exception as e:
        with task_lock:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['result'] = {'success': False, 'message': str(e)}
    finally:
        semaphore.release()
        dev_lock.release()

def start_analysis(device_id, config_overrides=None):
    task_id = str(uuid.uuid4())
    with task_lock:
        tasks[task_id] = {
            'status': 'pending',
            'result': None,
            'created_at': datetime.now(),
            'device_id': device_id,
            'db_id': None,
            'progress': None
        }
    threading.Thread(target=worker, args=(task_id, device_id, config_overrides), daemon=True).start()
    return task_id

def get_task_status(task_id):
    task = tasks.get(task_id)
    if task:
        resp = {
            'status': task['status'],
            'result': task['result'],
            'db_id': task.get('db_id'),
            'progress': task.get('progress')
        }
        return resp
    return None

def cleanup_old_tasks(max_age_seconds=3600):
    now = datetime.now()
    to_delete = []
    with task_lock:
        for tid, task in tasks.items():
            if (now - task['created_at']).total_seconds() > max_age_seconds:
                to_delete.append(tid)
        for tid in to_delete:
            del tasks[tid]