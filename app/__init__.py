# Disable matplotlib display
import matplotlib
matplotlib.use('Agg')
del matplotlib



from flask import Flask
from flask_bootstrap import Bootstrap
import os

from . import config

# Create flask application
app = Flask(__name__)

# Configure app
config = config.options[os.getenv('PIPELINE_CONFIG') or 'default']
app.config.from_object(config)

# Register extensions
bootstrap = Bootstrap(app)
if not (app.debug or app.testing or app.config['SSL_DISABLE']):
    from flask_sslify import SSLify
    sslify = SSLify(app)

# Register blueprints
from .main import main as main_blueprint
from .images import images as image_blueprint
app.register_blueprint(main_blueprint)
app.register_blueprint(image_blueprint, )