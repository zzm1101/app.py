# utils.py
# 全局工具函数：统一空值处理、数值转换、缓存键生成等

def normalize_product_item(value):
    """
    统一产品项的空值处理
    - None / 空字符串 / 纯空格 → None
    - 非空字符串 → 去除首尾空格后返回
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    try:
        s = str(value).strip()
        return s if s else None
    except:
        return None

def to_float_or_zero(value):
    """安全转换为 float，失败返回 0.0"""
    if value is None or value == '':
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def safe_division(numerator, denominator, default=0.0):
    """安全除法，分母为零时返回 default"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def get_cache_key(prefix, **kwargs):
    """生成带参数的缓存键"""
    if kwargs:
        items = [(k, v) for k, v in kwargs.items() if v is not None]
        if items:
            items.sort()
            param_str = '_'.join(f"{k}={v}" for k, v in items)
            return f"{prefix}_{param_str}"
    return prefix

def is_valid_date(date_obj, allow_future=False):
    """检查日期是否有效，默认不允许未来日期"""
    from datetime import date
    if date_obj is None:
        return False
    if not allow_future and date_obj > date.today():
        return False
    return True