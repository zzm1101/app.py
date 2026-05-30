# seal_monitor/utils.py
import sqlite3
import os
from flask import g, has_app_context
from config import Config

PARAM_LABELS = {
    'SEAL_VAE_SEQ_LEN': '序列长度',
    'SEAL_VAE_HIDDEN_DIM': '隐藏层维度',
    'SEAL_VAE_LATENT_DIM': '潜变量维度',
    'SEAL_VAE_EPOCHS': '训练轮数',
    'SEAL_VAE_BATCH_SIZE': '批大小',
    'SEAL_HEALTH_WARNING': '警告阈值',
    'SEAL_HEALTH_ALARM': '报警阈值',
    'SEAL_CIP_TEMPERATURE_THRESHOLD': 'CIP温度阈值(°C)',
    'SEAL_EMPTY_CURRENT_THRESHOLD': '空载电流阈值(A)',
}


def get_seal_db():
    """获取密封系统独立数据库连接"""
    if not has_app_context():
        db_path = Config.SEAL_DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    if 'seal_db' not in g:
        db_path = Config.SEAL_DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        g.seal_db = sqlite3.connect(db_path)
        g.seal_db.row_factory = sqlite3.Row
    return g.seal_db


def init_seal_db():
    """初始化密封系统数据库表"""
    db = get_seal_db()
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS analysis_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "device_id TEXT NOT NULL, "
            "analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "result_json TEXT, "
            "avg_health REAL, "
            "rul TEXT, "
            "anomaly_count INTEGER)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS device_config ("
            "device_id TEXT PRIMARY KEY, "
            "config_json TEXT NOT NULL, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS devices ("
            "device_id TEXT PRIMARY KEY, "
            "device_name TEXT NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cursor = db.execute('SELECT COUNT(*) FROM devices')
        if cursor.fetchone()[0] == 0:
            for device_name in ['设备01', '设备02', '设备03']:
                device_id = 'dev_' + device_name.replace(' ', '_')
                db.execute("INSERT OR IGNORE INTO devices (device_id, device_name) VALUES (?, ?)",
                           (device_id, device_name))
        db.commit()
    finally:
        if not has_app_context():
            db.close()