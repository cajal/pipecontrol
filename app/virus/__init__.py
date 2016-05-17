from flask import Blueprint

virus = Blueprint('virus', __name__)

from . import views
