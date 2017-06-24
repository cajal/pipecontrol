from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response, abort, session
from flask import Markup

from .forms import UserForm
from ..schemata import reso, experiment
from . import main

@main.route('/', methods=['GET', 'POST'])
def index():

    form = UserForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('User switched to {}'.format(form.user.data))
        session['user'] = form.user.data


    jobs = reso.schema.jobs & "user like '{}%%'".format(session['user'])
    return render_template('index.html',
                           user=session['user'],
                           form=form,
                           jobs=Markup(jobs._repr_html_()) if len(jobs) > 0 else None)

