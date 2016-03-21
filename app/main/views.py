from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response
from flask.ext.login import login_required, current_user
from flask.ext.sqlalchemy import get_debug_queries
from . import main
from .forms import EditProfileForm, EditProfileAdminForm
from .. import db
from ..models import Permission, User, Role, Schema
from ..decorators import admin_required, permission_required, guardian_required
from .errors import forbidden


@main.after_app_request
def after_request(response):
    for query in get_debug_queries():
        if query.duration >= current_app.config['ROWBOT_SLOW_DB_QUERY_TIME']:
            current_app.logger.warning(
                'Slow query: %s\nParameters: %s\nDuration: %fs\nContext: %s\n'
                % (query.statement, query.parameters, query.duration,
                   query.context))
    return response


@main.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)

    return render_template('index.html', user=current_user)


@main.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('user.html', user=user)


@main.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.name = form.name.data
        db.session.add(current_user)
        flash('Your profile has been updated.')
        return redirect(url_for('.user', username=current_user.username))
    form.name.data = current_user.name
    return render_template('edit_profile.html', form=form)

@main.route('/edit-profile/<username>', methods=['GET', 'POST'])
@login_required
@guardian_required
def edit_profile_admin(username):
    user = User.query.filter_by(username=username).first_or_404()
    form = EditProfileAdminForm(user=user, schemata=['{0}:{1}'.format(s.module, s.schema) for s in user.schemata])
    if form.validate_on_submit():
        user.email = form.email.data
        user.username = form.username.data
        user.confirmed = form.confirmed.data
        user.role = Role.query.get(form.role.data)
        user.name = form.name.data
        user.schemata = [Schema.query.filter_by(module=mod, schema=s).first()
                                for mod, s in map(lambda d: d.split(':'), form.schemata.data)]
        db.session.add(user)
        flash('The profile has been updated.')
        return redirect(url_for('.user', username=user.username))
    form.email.data = user.email
    form.username.data = user.username
    form.confirmed.data = user.confirmed
    form.role.data = user.role_id
    form.name.data = user.name


    return render_template('edit_profile.html', form=form, user=user)