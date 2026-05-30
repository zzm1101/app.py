# seal_monitor/__init__.py
from flask import Blueprint
import secrets

seal_bp = Blueprint('seal', __name__, url_prefix='/seal',
                    template_folder='templates',
                    static_folder='static')


def generate_csrf_token():
    """生成 CSRF token（用于模板）"""
    from flask import session
    if '_seal_csrf_token' not in session:
        session['_seal_csrf_token'] = secrets.token_hex(16)
    return session['_seal_csrf_token']


# 注意：不能在这里直接设置 jinja_env.globals，因为 blueprint 还没有 app 上下文
# 改为在注册蓝图后通过 app 设置，或者使用 @seal_bp.context_processor

@seal_bp.context_processor
def inject_csrf_token():
    """将 CSRF token 注入到模板上下文中"""
    return {'csrf_token': generate_csrf_token}


from . import routes