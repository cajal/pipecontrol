import sys
from collections import OrderedDict

from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response, abort, session
from flask import Markup
import datajoint as dj

from .decorators import ping
from .tables import CorrectionChannel, ProgressTable, SegmentationTask, JobTable
from .forms import UserForm, AutoProcessing, SummaryForm

from ..schemata import reso, experiment, shared, pupil, behavior
from . import main

import numpy as np

# -- bokeh
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3


@ping
@main.route('/')
def index():
    if not 'user' in session:
        return redirect(url_for('.user'))

    return render_template('index.html')


@ping
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


@ping
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


@ping
@main.route('/jobs', methods=['GET', 'POST'])
def jobs():
    schemas = OrderedDict(
        reso=reso,
        behavior=behavior,
        pupil=pupil
    )

    if request.method == 'POST':
        to_delete = [dict(key_hash=e) for e in request.form.getlist('delete_item')]
        schema = request.form['schema']
        if schemas[schema].schema.jobs & to_delete & 'status="reserved"':
            flash('Though shalt not delete reserved jobs')
        rel = schemas[schema].schema.jobs & to_delete & 'status="error"'
        n = len(rel)
        rel.delete()
        flash('{} entries deleted.'.format(n))
        return redirect(url_for('.jobs'))

    kwargs = {}
    if request.method == 'GET' and 'sort' in request.args:
        kwargs = dict(order_by='{sort} {direction}'.format(**request.args))

    all_jobs = {}
    for key, schema in schemas.items():
        jobs = schema.schema.jobs.proj('table_name', 'status', 'key_hash',
                                       'error_message', 'key', 'timestamp').fetch(as_dict=True, **kwargs)
        for r in jobs:
            r['delete'] = ('delete_item', r['key_hash'])
        jobs = JobTable(jobs, target='main.jobs', exlude=['key', 'delete'])
        all_jobs[key] = jobs
    return render_template('jobs.html', jobs=all_jobs)


@ping
@main.route('/progress', methods=['GET', 'POST'])
def progress():
    tmp = {e: getattr(reso, e)().progress(experiment.Session() & 'username="{}"'.format(session['user'])) \
           for e in dir(reso) if not e.startswith('_') \
           and not e == 'schema' \
           and issubclass(getattr(reso, e), dj.Computed)}
    progress = [dict(relation=e, finished=v[0], total=v[1],
                     percent='{:.1f}%'.format(v[0] / v[1] * 100 if v[1] > 0 else 100, 1))
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


@ping
@main.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    compartment_prefix = 'compartment'
    select_prefix = 'select'
    exclude_prefix = 'exclude'

    info = (reso.MotionCorrection() * reso.ScanInfo()).proj('nchannels')
    jobs = (info * shared.Channel() & 'channel <= nchannels') \
           - reso.SegmentationTask() - reso.DoNotSegment() \
           & (experiment.Session() & "username='{}'".format(session['user']))
    pk = jobs.heading.primary_key
    djkeys = jobs.fetch(dj.key)

    if request.method == 'POST':
        skeys = [_encode(k, pk) for k in djkeys]
        keys = [dict(_decode(s, pk),
                     compartment=request.form[compartment_prefix + s],
                     segmentation_method=2)
                for s in skeys if request.form.get(select_prefix + s) \
                and not request.form.get(exclude_prefix + s)]
        nkeys = [_decode(s, pk) for s in skeys if request.form.get(exclude_prefix + s)]

        reso.SegmentationTask().insert(keys, ignore_extra_fields=True)
        flash('{} keys inserted into SegmentationTask.'.format(len(keys)))

        reso.DoNotSegment().insert(nkeys, ignore_extra_fields=True)
        flash('{} excluded.'.format(len(nkeys)))
        return redirect(url_for('.segmentation'))

    compartments = experiment.Compartment().fetch('compartment')
    keys = [dict(k,
                 compartment=(_encode(k, pk, compartment_prefix), compartments, 'soma'),
                 select=_encode(k, pk, select_prefix),
                 exclude=_encode(k, pk, exclude_prefix)
                 ) for k in djkeys]
    table = SegmentationTask(keys)
    return render_template('segmentation.html',
                           table=table)


@ping
@main.route('/summary', methods=['GET', 'POST'])
def summary():
    form = SummaryForm(request.form)
    figure = None
    if request.method == 'POST' and form.validate():
        key = dict(
            animal_id=form['animal_id'].data,
            session=form['session'].data,
            scan_idx=form['scan_idx'].data,
            slice=form['slice'].data,
        )
        if reso.SummaryImages() & key:
            # dict(animal_id=8623, slice=1, session=1, scan_idx=15)
            avg, corr, chan = (reso.SummaryImages() & key).fetch(
                'average', 'correlation', 'channel')

            Ic = np.zeros(corr[0].shape + (3,))
            Ia = np.zeros_like(Ic)

            ch2ch = {1: 1, 2: 0}
            for imgc, imga, c in zip(corr, avg, chan):
                Ic[..., ch2ch[int(c)]] = imgc.squeeze()
                Ia[..., ch2ch[int(c)]] = imga.squeeze()
            Ic = (Ic - Ic.min()) / (Ic.max() - Ic.min())
            Ia = (Ia - Ia.min()) / (Ia.max() - Ia.min())
            fig, ax = plt.subplots(1, 2, figsize=(12, 10))
            ax[0].imshow(Ic, origin='lower', interpolation='nearest')
            ax[0].set_title('Correlation Image')
            ax[1].imshow(Ia, origin='lower', interpolation='nearest')
            ax[1].set_title('Average Image')
            [a.axis('off') for a in ax.ravel()]
            figure = mpld3.fig_to_html(fig)
        else:
            flash('Could not find figure for key {}'.format(str(key)))

    return render_template('summary.html', figure=figure, form=form)
