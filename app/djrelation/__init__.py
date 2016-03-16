from flask import Blueprint
import datajoint as dj

PER_PAGE = 20
ENABLE_EDIT=True


djpage = Blueprint('djrelation', __name__)
#
from . import views
