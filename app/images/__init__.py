from flask import Blueprint

images = Blueprint('images', __name__,  url_prefix="/images")

# Create views and error handlers
from . import views