import inspect
from datetime import datetime
import hashlib
from importlib import import_module
import os
from fabric.utils import abort
from sqlalchemy import UniqueConstraint

from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer, BadSignature
from markdown import markdown
import bleach
from flask import current_app, request, url_for
from flask.ext.login import UserMixin, AnonymousUserMixin
from app.exceptions import ValidationError
from . import db, login_manager
import datajoint as dj



class Permission:
    READ = 0x01
    WRITE = 0x02
    GRANT = 0x04
    ADMINISTER = 0x80


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(64), unique=True)
    default = db.Column(db.Boolean, default=False, index=True)
    permissions = db.Column(db.Integer)
    users = db.relationship('User', backref='role', lazy='dynamic')

    @staticmethod
    def insert_roles():
        roles = {
            'Viewer': (Permission.READ , True),
            'User': (Permission.READ | Permission.WRITE, True),
            'Moderator': (Permission.READ | Permission.WRITE | Permission.GRANT, False),
            'Administrator': (0xff, False)
        }
        for r in roles:
            role = Role.query.filter_by(name=r).first()
            if role is None:
                role = Role(name=r)
            role.permissions, role.default = roles[r]
            db.session.add(role)
        db.session.commit()

    def __repr__(self):
        return '<Role %r>' % self.name

user_schema_access = db.Table('user_schema_access',
                              db.Column('user_id', db.Integer, db.ForeignKey('users.id'), nullable=False ),
                              db.Column('module_name', db.String(256), nullable=False),
                              db.Column('schema_name', db.String(128), nullable=False),
                              db.ForeignKeyConstraint(('module_name', 'schema_name'),
                                                      ('schemata.module', 'schemata.schema')),
                              db.PrimaryKeyConstraint('user_id', 'module_name', 'schema_name'),
                              )


class Schema(db.Model):
    __tablename__ = 'schemata'

    module = db.Column(db.String(256), primary_key=True)
    schema = db.Column(db.String(128), primary_key=True)
    users = db.relationship('User', secondary=user_schema_access, backref='schemata')

    @staticmethod
    def insert_schemata():
        schemas = os.getenv('DJ_MODS')
        for mod_name in schemas.split(':'):
            mod = import_module(mod_name)
            for name, _ in inspect.getmembers(mod,
                                lambda k: isinstance(k, type) and issubclass(k, (dj.Manual, dj.Lookup))):
                s = Schema.query.filter_by(module=mod_name, schema=name).first()
                if s is None:
                    s = Schema(module=mod_name, schema=name)
                for admin in User.query.join(User.role, aliased=True).filter_by(name='Administrator'):
                    if admin not in s.users:
                        s.users.append(admin)
                db.session.add(s)

        db.session.commit()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    name = db.Column(db.String(64))

    email = db.Column(db.String(64), unique=True, index=True)

    password_hash = db.Column(db.String(128))
    confirmed = db.Column(db.Boolean, default=False)

    member_since = db.Column(db.DateTime(), default=datetime.utcnow)
    last_seen = db.Column(db.DateTime(), default=datetime.utcnow)

    avatar_hash = db.Column(db.String(32))

    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if self.email is not None and self.avatar_hash is None:
            self.avatar_hash = hashlib.md5(
                self.email.encode('utf-8')).hexdigest()

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_confirmation_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'confirm': self.id})

    def confirm(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return False
        if data.get('confirm') != self.id:
            return False
        self.confirmed = True
        db.session.add(self)
        return True

    @staticmethod
    def user_from_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except BadSignature:
            abort(403)
        user = User.query.filter_by(id=data['confirm']).first_or_404()
        return user

    @staticmethod
    def confirm_user(token):
        user = User.user_from_token(token)
        return user.confirm(token)

    def generate_reset_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'reset': self.id})

    def reset_password(self, token, new_password):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return False
        if data.get('reset') != self.id:
            return False
        self.password = new_password
        db.session.add(self)
        return True

    def generate_email_change_token(self, new_email, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'change_email': self.id, 'new_email': new_email})

    def change_email(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return False
        if data.get('change_email') != self.id:
            return False
        new_email = data.get('new_email')
        if new_email is None:
            return False
        if self.query.filter_by(email=new_email).first() is not None:
            return False
        self.email = new_email
        self.avatar_hash = hashlib.md5(
            self.email.encode('utf-8')).hexdigest()
        db.session.add(self)
        return True

    def can(self, permissions):
        return self.role is not None and \
               (self.role.permissions & permissions) == permissions

    def is_administrator(self):
        return self.can(Permission.ADMINISTER)

    def ping(self):
        self.last_seen = datetime.utcnow()
        db.session.add(self)

    def gravatar(self, size=100, default='identicon', rating='g'):
        if request.is_secure:
            url = 'https://secure.gravatar.com/avatar'
        else:
            url = 'http://www.gravatar.com/avatar'
        hash = self.avatar_hash or hashlib.md5(
            self.email.encode('utf-8')).hexdigest()
        return '{url}/{hash}?s={size}&d={default}&r={rating}'.format(
            url=url, hash=hash, size=size, default=default, rating=rating)

    def __repr__(self):
        return '<User %r>' % self.username


class AnonymousUser(AnonymousUserMixin):
    def can(self, permissions):
        return False

    def is_administrator(self):
        return False







login_manager.anonymous_user = AnonymousUser

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
