from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response
from flask.ext.login import login_required, current_user
from flask.ext.sqlalchemy import get_debug_queries
from flask_weasyprint import HTML, render_pdf, CSS

from app.decorators import admin_required
from .forms import SelectMouseForm
from . import mouse
from .. import db
from commons import mice, virus
from weasyprint import HTML
import numpy as np
from fabee import inj

@mouse.route('/<animal_id>')
@login_required
def index(animal_id):
    return redirect(url_for('.cagecard', animal_id=animal_id))

@mouse.route('/cagecard/', methods=['GET', 'POST'])
@login_required
def select_mouse():
    form = SelectMouseForm()
    if form.validate_on_submit():
        if form.pdf.data:
            return redirect(url_for('.cagecard_pdf', animal_id=form.animal_id.data))
        else:
            return redirect(url_for('.cagecard', animal_id=form.animal_id.data))
    return render_template('mouse/select_mouse.html', form=form)


@mouse.route('/cagecard/<animal_id>')
@login_required
def cagecard(animal_id):
    info = (mice.Mice() & dict(animal_id=animal_id)).fetch1()
    info['lines'] = (mice.Genotypes() & dict(animal_id=animal_id)).fetch.as_dict()

    # find out parents
    parent_ids = [int(i) for i in (mice.Parents() & dict(animal_id=animal_id)).fetch['parent_id']]
    parent_rstr = [{'animal_id':p } for p in parent_ids]
    parents = (mice.Mice() & parent_rstr).fetch.as_dict()

    parents2 = {}

    for p in parents:
        if p['sex'] in parents2:
            parents2[p['sex']] += ', {0}'.format(p['animal_id'])
        else:
            parents2[p['sex']] = str(p['animal_id'])
        parent_ids.remove(p['animal_id'])
    info['parents'] = parents2

    injections =  inj.Injection()*inj.Substance()*inj.Substance.Virus()*virus.Virus() & dict(animal_id=animal_id)
    if injections:
        info['injections'] = injections.proj('area','construct_id','toi','virus_lot').fetch.as_dict()
        for i in info['injections']:
            i['toi'] = str(i['toi']).split()[0]

    return render_template('mouse/cagecard.html', protocol='AN-4703', **info)


@mouse.route('/cagecard/<animal_id>.pdf')
@login_required
def cagecard_pdf(animal_id):
    # Make a PDF from another view
    html = cagecard(animal_id=animal_id)
    print(html)
    return render_pdf(HTML(string=html),
                      stylesheets=[CSS(url_for('static', filename='mouse/style.css')),
                                   CSS('//netdna.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css')])
