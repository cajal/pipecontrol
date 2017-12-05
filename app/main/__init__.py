from flask import Blueprint

main = Blueprint('main', __name__)

# Create views and error handlers
from . import views, errors