from flask import Flask
from flask_bootstrap import Bootstrap
from config import config
from flask_qrcode import QRcode

bootstrap = Bootstrap()
def create_app(config_name):
    app = Flask(__name__)
    QRcode(app)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    bootstrap.init_app(app)

    if not app.debug and not app.testing and not app.config['SSL_DISABLE']:
        from flask.ext.sslify import SSLify
        sslify = SSLify(app)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)


    return app
