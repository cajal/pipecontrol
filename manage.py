#!/usr/bin/env python
import getpass
import os
from config import Config



# --- import extensions and apps
from app import create_app
from flask_script import Manager, Server

# -- create app and register with extensions
app = create_app(os.getenv('PIPELINE_CONFIG') or 'default')
manager = Manager(app)

manager.add_command('runserver', Server(host="0.0.0.0"))


if __name__ == '__main__':
    manager.run()
