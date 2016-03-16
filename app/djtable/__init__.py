from flask import Blueprint

PER_PAGE = 20
ENABLE_EDIT=True


djpage = Blueprint('djtable', __name__)
#
from . import views


