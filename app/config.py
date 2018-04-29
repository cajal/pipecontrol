import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    SSL_DISABLE = True

class DevelopmentConfig(Config):
    DEBUG = True
    SERVER_NAME='localhost'

class ProductionConfig(Config):
    DEBUG = False # this won't work with flask script, use Flask.run() instead
    SERVER_NAME='shikigami.ad.bcm.edu'

options = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
