from flask import Blueprint

mouse = Blueprint('mouse', __name__)

from . import views
