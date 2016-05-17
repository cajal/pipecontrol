from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response
from flask.ext.login import login_required, current_user
from flask.ext.sqlalchemy import get_debug_queries

from app.decorators import admin_required
from .forms import NewVirusForm
from . import virus
from .. import db



@virus.route('/enter', methods=['GET', 'POST'])
@login_required
def enter():
    form = NewVirusForm()
    if form.validate_on_submit():
        form.enter()
        return redirect(url_for('djtable.display', relname='commons.virus.Virus'))
    return render_template('virus/enter_virus.html', form=form)


