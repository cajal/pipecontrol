#!/usr/bin/env python
import os

# --- import environment variables from hidden file
if os.path.exists('.env'):
    print('Importing environment from .env...')
    for line in open('.env'):
        var = line.strip().split('=')
        if len(var) == 2:
            os.environ[var[0]] = var[1]

# --- import extensions and apps
from app import create_app, db
from app.models import User,  Permission,  Role
from flask.ext.script import Manager, Shell
from flask.ext.migrate import Migrate, MigrateCommand
# -- create app and register with extensions
app = create_app(os.getenv('ROWBOT_CONFIG') or 'default')
manager = Manager(app)
migrate = Migrate(app, db)

# --- shell context
def make_shell_context():
    return dict(app=app, db=db, User=User, Role=Role, Permission=Permission)
manager.add_command("shell", Shell(make_context=make_shell_context))


# --- database and migration
manager.add_command('db', MigrateCommand)


# --- deployment command

@manager.command
def deploy():
    """Run deployment tasks."""
    from flask.ext.migrate import upgrade
    from app.models import User

    # migrate database to latest revision
    upgrade()


    # create self-follows for all users
    User.feed_to_self()


# --- run the application

@manager.command
def init_dev():
    """Initialize database, migrate, upgrade, and perform initial inserts."""
    from flask.ext.migrate import upgrade, init, migrate
    from app.models import Role, User

    init()
    migrate()
    upgrade()

    # create user roles
    Role.insert_roles()

    u = User(username='admin', email='admin@rowbot.org', confirmed=True)
    u.password = 'test123'
    u.role_id = Role.query.filter_by(name='Administrator').first().id
    db.session.add(u)

if __name__ == '__main__':

    manager.run()
