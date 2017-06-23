from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response, abort, session
from flask import Markup
from ..schemata import reso
from . import main

@main.route('/', methods=['GET', 'POST'])
def index():
    jobs = reso.schema.jobs
    return render_template('index.html', user='jake', jobs=Markup(jobs._repr_html_()))



