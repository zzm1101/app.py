# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ===== 原系统常量（模块级，供各模块直接导入） =====
DAIRY_GB = {
    "protein_min": 2.9,
    "acidity_min": 70,
    "acidity_max": 85,
    "coliform_max": 1,
    "mold_yeast_max": 30,
    "viable_bacteria_min": 1e6,
    "net_content_tolerance": 4.5,
    "shelf_life_days": 21,
    "cold_chain_std_temp": 4,
    "fermentation_std_hours": 6
}

FMEA_SEVERITY_K = {
    10: {"a0_coef": 10.0, "hidden_coef": 5.0, "risk_level": "极高风险", "desc": "危及安全/合规"},
    9: {"a0_coef": 8.0, "hidden_coef": 4.0, "risk_level": "高风险", "desc": "批量召回"},
    8: {"a0_coef": 6.0, "hidden_coef": 3.0, "risk_level": "高风险", "desc": "批量客户投诉"},
    7: {"a0_coef": 4.0, "hidden_coef": 2.5, "risk_level": "中高风险", "desc": "批次返工"},
    6: {"a0_coef": 2.0, "hidden_coef": 2.0, "risk_level": "中风险", "desc": "挑选返工"},
    5: {"a0_coef": 1.5, "hidden_coef": 1.5, "risk_level": "中风险", "desc": "轻微返工"},
    4: {"a0_coef": 1.2, "hidden_coef": 1.2, "risk_level": "低风险", "desc": "不影响功能"},
    3: {"a0_coef": 1.0, "hidden_coef": 1.1, "risk_level": "低风险", "desc": "外观轻微瑕疵"},
    2: {"a0_coef": 0.8, "hidden_coef": 1.0, "risk_level": "极低风险", "desc": "无影响"},
    1: {"a0_coef": 0.5, "hidden_coef": 1.0, "risk_level": "无风险", "desc": "无任何影响"}
}

FEATURE_TYPE = {
    "nominal": {"name": "望目特性", "formula": "L(y) = k·(y-m)²", "desc": "存在固定目标值"},
    "smaller": {"name": "望小特性", "formula": "L(y) = k·y²", "desc": "数值越小越好"},
    "larger": {"name": "望大特性", "formula": "L(y) = k/y²", "desc": "数值越大越好"}
}

TARGET_METHODS = {
    "method_1": {"name": "国标约束保底法", "desc": "国标规格限中值"},
    "method_2": {"name": "PPK过程能力反推法", "desc": "确保99.99%批次合规"},
    "method_3": {"name": "SN信噪比稳健优化法", "desc": "最大化抗干扰能力"},
    "method_4": {"name": "PPK+SN联合优化法", "desc": "兼顾合规与稳健"},
    "method_5": {"name": "期望损失最小化法", "desc": "全局最小化质量损失"},
    "method_6": {"name": "Arrhenius货架期衰减法", "desc": "酸奶专属保质期优化"},
    "method_7": {"name": "VOC客户反馈法", "desc": "暂未开放"}
}

PAF_CATEGORY = {
    "prevention": "预防成本",
    "appraisal": "鉴定成本",
    "internal_failure": "内部故障成本",
    "external_failure": "外部故障成本"
}


class Config:
    # ===== 原系统配置 =====
    SECRET_KEY = os.environ.get('SECRET_KEY', 'yogurt_qlf_default_dev_key_change_me')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///yogurt_qlf.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_AS_ASCII = False
    PER_PAGE = 20
    SEND_FILE_MAX_AGE_DEFAULT = 3600

    # 缓存配置
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'SimpleCache')
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', None)
    CACHE_DEFAULT_TIMEOUT = 60

    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 20 * 1024 * 1024))  # 20MB
    FETCH_BATCH_SIZE = 1000
    MAX_BATCHES_FOR_SPC = 50
    MAX_SAMPLES_FOR_SPC = 5000

    # 性能开关
    CACHE_WARMUP = os.environ.get('CACHE_WARMUP', 'true').lower() == 'true'

    # 机器学习影响分析模块开关
    ENABLE_ML_INFLUENCE = os.environ.get('ENABLE_ML_INFLUENCE', 'false').lower() == 'true'

    # ===== 密封监测系统配置 =====
    SEAL_DB_PATH = os.environ.get('SEAL_DB_PATH', os.path.join(BASE_DIR, 'seal_history.db'))
    SEAL_DATA_DIR = os.environ.get('SEAL_DATA_DIR', os.path.join(BASE_DIR, 'seal_data'))
    SEAL_MODEL_DIR = os.environ.get('SEAL_MODEL_DIR', os.path.join(BASE_DIR, 'seal_models'))

    SEAL_MAX_CONCURRENT_TASKS = int(os.environ.get('SEAL_MAX_CONCURRENT_TASKS', 3))

    SEAL_HEALTH_NORMAL = float(os.environ.get('SEAL_HEALTH_NORMAL', 0.8))
    SEAL_HEALTH_WARNING = float(os.environ.get('SEAL_HEALTH_WARNING', 0.5))
    SEAL_HEALTH_ALARM = float(os.environ.get('SEAL_HEALTH_ALARM', 0.3))

    SEAL_CIP_TEMPERATURE_THRESHOLD = float(os.environ.get('SEAL_CIP_TEMPERATURE_THRESHOLD', 70))
    SEAL_EMPTY_CURRENT_THRESHOLD = float(os.environ.get('SEAL_EMPTY_CURRENT_THRESHOLD', 8))
    SEAL_TRANSITION_DURATION_MIN = int(os.environ.get('SEAL_TRANSITION_DURATION_MIN', 30))

    SEAL_VAE_SEQ_LEN = int(os.environ.get('SEAL_VAE_SEQ_LEN', 30))
    SEAL_VAE_HIDDEN_DIM = int(os.environ.get('SEAL_VAE_HIDDEN_DIM', 64))
    SEAL_VAE_LATENT_DIM = int(os.environ.get('SEAL_VAE_LATENT_DIM', 16))
    SEAL_VAE_EPOCHS = int(os.environ.get('SEAL_VAE_EPOCHS', 80))
    SEAL_VAE_BATCH_SIZE = int(os.environ.get('SEAL_VAE_BATCH_SIZE', 64))

    SEAL_WINDOW_SIZE = int(os.environ.get('SEAL_WINDOW_SIZE', 10))

    SEAL_MAX_CONTENT_LENGTH = int(os.environ.get('SEAL_MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
    SEAL_ALLOWED_EXTENSIONS = {'csv'}

    SEAL_SENSOR_COLUMNS = {
        'required': ['timestamp'],
        'numeric': ['ae_energy', 'ae_count', 'vibration_rms', 'temperature', 'current', 'pressure'],
        'optional': []
    }

    SEAL_CONDITION_COLUMNS = {
        'cip_temp': 'temperature',
        'empty_current': 'current'
    }

    SEAL_FEATURE_RECIPES = {
        'ae_energy': ['mean', 'std', 'wavelet_entropy'],
        'ae_count': ['mean'],
        'vibration_rms': ['mean', 'kurtosis'],
        'temperature': ['mean', 'gradient'],
        'current': ['mean'],
        'pressure': ['diff_abs'],
    }

    @classmethod
    def get_with_overrides(cls, overrides: dict):
        """动态生成配置对象，支持运行时覆盖参数（用于密封系统异步任务）"""
        new_config = {}
        for key in dir(cls):
            if not key.startswith('_') and isinstance(getattr(cls, key), (int, float, str, bool, list, dict)):
                new_config[key] = overrides.get(key, getattr(cls, key))
        return type('DynamicConfig', (), new_config)


# 创建必要目录
for path in [Config.SEAL_DATA_DIR, Config.SEAL_MODEL_DIR]:
    os.makedirs(path, exist_ok=True)