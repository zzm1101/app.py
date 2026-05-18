# extensions.py
from flask_caching import Cache

cache = Cache()

CACHE_KEYS = {
    'DASHBOARD': 'dashboard_v2',
    'LOSS_ANALYSIS': 'loss_analysis_v2',
    'IMPROVEMENT': 'improvement_v2',
    'PRODUCTION_LIST': 'production_list_v2',
}

def clear_all_caches():
    for key in CACHE_KEYS.values():
        cache.delete(key)
    cache.delete_many('dashboard_view', 'loss_analysis_view', 'improvement_view', 'production_list_view')