import sys
from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response, abort, session
from flask import Markup
import datajoint as dj

from .decorators import ping
from .tables import CorrectionChannel, ProgressTable
from .forms import UserForm, AutoProcessing

from ..schemata import reso, experiment, shared
from . import main


@main.route('/', methods=['GET', 'POST'])
def index():
    form = UserForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('User switched to {}'.format(form.user.data))
        session['user'] = form.user.data
    else:
        form.user.data = session['user']

    return render_template('index.html',
                           form=form)


@main.route('/autoprocessing', methods=['GET', 'POST'])
def autoprocessing():
    form = AutoProcessing(request.form)
    if request.method == 'POST' and form.validate():
        key = dict(
            animal_id=form['animal_id'].data,
            session=form['session'].data,
            scan_idx=form['scan_idx'].data,
        )
        if experiment.AutoProcessing() & key:
            flash('Key {} is already in  AutoProcessing'.format(str(key)))
        else:
            experiment.AutoProcessing().insert1(key)
            flash('Key {} has been inserted into AutoProcessing'.format(str(key)))

    return render_template('autoprocessing.html',
                           form=form)


@main.route('/user/', methods=['GET', 'POST'], defaults={'username': None})
@main.route('/user/<username>')
def user(username):
    form = UserForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('User switched to {}'.format(form.user.data))
        session['user'] = form.user.data
    elif 'user' in session:
        form.user.data = session['user']
    else:
        session['user'] = 'unknown'

    if username is not None:
        session['user'] = username
        flash('User switched to {}'.format(username))

    return render_template('user.html', form=form)


@main.route('/jobs', methods=['GET', 'POST'])
def jobs():
    jobs = reso.schema.jobs & "user like '{}%%'".format(session['user'])

    return render_template('jobs.html',
                           jobs=Markup(jobs._repr_html_()) if len(jobs) > 0 else None)


@main.route('/progress', methods=['GET', 'POST'])
def progress():
    tmp = {e: getattr(reso, e)().progress(experiment.Session() & 'username="{}"'.format(session['user'])) \
           for e in dir(reso) if not e.startswith('_') \
           and not e == 'schema' \
           and issubclass(getattr(reso, e), dj.Computed)}
    progress = [dict(relation=e, finished=v[0], total=v[1], percent='{:.1f}%'.format(v[0] / v[1] * 100, 1))
                for e, v in tmp.items()]
    table = ProgressTable(progress)
    return render_template('progress.html', table=table)


def _encode(key, primary_key, prefix=''):
    template = prefix + '-'.join(['{{{}}}'.format(k) for k in primary_key])
    return template.format(**key)


def _decode(s, primary_key, prefix=''):
    return dict(zip(primary_key, map(int, s[len(prefix):].split('-'))))


@ping
@main.route('/correction', methods=['GET', 'POST'])
def correction():
    channel_prefix = 'channel'
    select_prefix = 'select'

    scaninfo = (reso.ScanInfo().proj('nslices', 'nchannels') * shared.Slice() & 'slice <= nslices') \
               - reso.CorrectionChannel() \
               & (experiment.Session() & "username='{}'".format(session['user']))
    pk = scaninfo.heading.primary_key
    djkeys, channels = scaninfo.fetch(dj.key, 'nchannels')

    if request.method == 'POST':
        skeys = [_encode(k, pk) for k in djkeys]
        keys = [dict(_decode(s, pk), channel=int(request.form[channel_prefix + s]))
                for s in skeys if request.form.get(select_prefix + s)]
        reso.CorrectionChannel().insert(keys, ignore_extra_fields=True)
        flash('{} keys inserted.'.format(len(keys)))
        return redirect(url_for('.correction'))

    keys = [dict(k,
                 channel=(c, _encode(k, pk, channel_prefix)),
                 select=_encode(k, pk, select_prefix)) for k, c in zip(djkeys, channels)]
    table = CorrectionChannel(keys)
    return render_template('correction.html',
                           table=table)
