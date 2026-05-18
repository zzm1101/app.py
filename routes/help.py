from flask import Blueprint, render_template

help_bp = Blueprint('help', __name__, url_prefix='/help')

@help_bp.route('/')
def help_index():
    return render_template('help.html')