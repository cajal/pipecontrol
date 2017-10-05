import sys
from collections import OrderedDict
from functools import partial

from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, make_response, abort, session, send_from_directory
from flask import Markup
from scipy.signal import convolve2d

from .. import schemata
import datajoint as dj
from datajoint.base_relation import lookup_class_name
from datajoint.erd import _get_tier
from .utils import namehash
from .decorators import ping
from .tables import ResoCorrectionTable, ProgressTable, JobTable, SummaryTable, ChannelCol, \
    MesoCorrectionTable, MesoSegmentationTask, ResoSegmentationTask, djtable
from .forms import UserForm, AutoProcessing, SummaryForm, RestrictionForm, TrackingForm

from ..schemata import reso, experiment, shared, pupil, behavior, meso
from . import main

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins, utils
from graphviz import Digraph


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
@main.route('/tracking/<animal_id>/<session>/<scan_idx>', methods=['GET', 'POST'])
def tracking(animal_id, session, scan_idx):
    form = TrackingForm(request.form)
    key = dict(
        animal_id=animal_id,
        session=session,
        scan_idx=scan_idx,
    )
    figure = None
    if pupil.Eye() & key:
        prev = (pupil.Eye() & key).fetch1('preview_frames')

        fig, ax = plt.subplots(4,4,figsize=(10, 8), sharex="col", sharey="row")
        for a, fr in zip(ax.ravel(), prev.transpose([2,0,1])):
            a.imshow(fr, cmap='gray', interpolation='bicubic')
            a.axis('off')
            a.set_aspect(1)
        plugins.connect(fig, plugins.LinkedBrush([])) # TODO Edgar change that here
        figure = mpld3.fig_to_html(fig)
    else:
        flash('Could not find figure for key {}'.format(str(key)))
    if request.method == 'POST' and form.validate():
        pass

    return render_template('trackingtask.html',
                           form=form, figure=figure)



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
        pupil=pupil,
        meso=meso
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
                                       'error_message', 'connection_id', 'user',
                                       'key', 'timestamp').fetch(as_dict=True, **kwargs)
        for r in jobs:
            r['delete'] = ('delete_item', r['key_hash'])
        jobs = JobTable(jobs, target='main.jobs', exlude=['key', 'delete'])
        all_jobs[key] = jobs
    return render_template('jobs.html', jobs=all_jobs)


@ping
@main.route('/progress', methods=['GET', 'POST'])
def progress():
    tmp = {e: getattr(reso, e)().progress(experiment.Session() & 'username="{}"'.format(session['user']), display=False) \
           for e in dir(reso) if not e.startswith('_') \
           and not e == 'schema' \
           and issubclass(getattr(reso, e), dj.Computed)}
    progress = [dict(relation=e, remaining=v[0], total=v[1],
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
    tables = {}
    djkeys, pk = {}, {}
    schemata = OrderedDict([('reso', reso), ('meso', meso)])

    for (name, schema), (field, Field) in zip(schemata.items(), zip(['slice', 'field'], [shared.Slice, shared.Field])):
        nfield = 'n{}s'.format(field)
        # -- encode reso table
        scaninfo = (schema.ScanInfo().proj(nfield, 'nchannels') * Field() & '{} <= {}'.format(field, nfield)) \
                   - schema.CorrectionChannel() \
                   & (experiment.Session() & "username='{}'".format(session['user']))

        pk[name] = scaninfo.heading.primary_key
        djkeys[name], channels = scaninfo.fetch(dj.key, 'nchannels')
        keys = [dict(k,
                     channel=(c, _encode(k, pk[name], channel_prefix)),
                     select=('selected', _encode(k, pk[name]))) for k, c in zip(djkeys[name], channels)]
        if name == 'reso':
            table = ResoCorrectionTable(keys)
        else:
            table = MesoCorrectionTable(keys)
        tables[name] = table

    if request.method == 'POST':
        name = request.form['schema']
        schema = schemata[name]
        skeys = [_encode(k, pk[name]) for k in djkeys[name]]

        selected = request.form.getlist('selected')
        keys = [dict(_decode(s, pk[name]), channel=int(request.form[channel_prefix + s])) for s in skeys if
                s in selected]
        schema.CorrectionChannel().insert(keys, ignore_extra_fields=True)
        flash('{} keys inserted into {}.CorrectionChannel.'.format(len(keys), name))
        return redirect(url_for('.correction'))  # redirect to reload the tables

    return render_template('correction.html', tables=tables)


@ping
@main.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    compartment_prefix = 'compartment'
    select_prefix = 'select'
    exclude_prefix = 'exclude'
    tables = {}
    djkeys, pk = {}, {}

    schemata = OrderedDict([('reso', reso), ('meso', meso)])
    compartments = experiment.Compartment().fetch('compartment')
    for name, schema in schemata.items():
        info = (schema.MotionCorrection() * schema.ScanInfo()).proj('nchannels')
        jobs = (info * shared.Channel() & 'channel <= nchannels') \
               - schema.SegmentationTask() - schema.DoNotSegment() \
               & (experiment.Session() & "username='{}'".format(session['user']))
        pk[name] = jobs.heading.primary_key
        djkeys[name] = jobs.fetch(dj.key)

        keys = [dict(k,
                     compartment=(_encode(k, pk[name], compartment_prefix), compartments, 'soma'),
                     select=('selected', _encode(k, pk[name])),
                     exclude=('excluded', _encode(k, pk[name])),
                     ) for k in djkeys[name]]
        if name == 'reso':
            table = ResoSegmentationTask(keys)
        else:
            table = MesoSegmentationTask(keys)
        tables[name] = table

    if request.method == 'POST':
        name = request.form['schema']
        schema = schemata[name]
        skeys = [_encode(k, pk[name]) for k in djkeys[name]]
        selected = request.form.getlist('selected')
        excluded = request.form.getlist('excluded')
        keys = [dict(_decode(s, pk[name]),
                     compartment=request.form[compartment_prefix + s],
                     segmentation_method=3)
                for s in skeys if s in selected and not s in excluded]
        nkeys = [_decode(s, pk[name]) for s in excluded]

        schema.SegmentationTask().insert(keys, ignore_extra_fields=True)
        flash('{} keys inserted into {}.SegmentationTask.'.format(len(keys), name))

        schema.DoNotSegment().insert(nkeys, ignore_extra_fields=True)
        flash('{} excluded in {}.'.format(len(nkeys), name))
        return redirect(url_for('.segmentation'))

    return render_template('segmentation.html',
                           tables=tables)


@ping
@main.route('/summary', methods=['GET', 'POST'])
def summary():
    form = RestrictionForm(request.form)
    restriction = None
    # figure = None
    if request.method == 'POST' and form.validate():
        restriction = form['restriction'].data

    if restriction is not None:
        content = (reso.SummaryImages() & restriction).proj().fetch(as_dict=True, limit=40)
    else:
        content = reso.SummaryImages().proj().fetch(as_dict=True, limit=40)

    for c in content:
        c['correlation'] = url_for('main.summary_image', which='correlation', **c)
        c['average'] = url_for('main.summary_image', which='average', **c)
        if reso.Activity() & c:
            c['trace'] = url_for('main.traces',
                                 **(reso.Activity() & c & dict(segmentation_method=3, spike_method=5)).fetch1(dj.key))
        else:
            c['trace'] = None

    table = SummaryTable(content)

    return render_template('summary.html', form=form, table=table)


@ping
@main.route('/image/<animal_id>/<session>/<scan_idx>/<slice>/<reso_version>/<which>')
def summary_image(animal_id, session, scan_idx, slice, reso_version, which):
    key = dict(
        animal_id=animal_id, slice=slice, session=session, scan_idx=scan_idx, reso_version=reso_version
    )
    figure = None
    if reso.SummaryImages() & key:
        corr, chan = (reso.SummaryImages() \
                      * reso.SummaryImages.Average() \
                      * reso.SummaryImages.Correlation() & key).fetch('{}_image'.format(which), 'channel')

        I = np.zeros(corr[0].shape + (3,))

        ch2ch = {1: 1, 2: 0}
        for img, c in zip(corr, chan):
            if which == 'average':
                img = img.squeeze()
                h = np.hamming(51)
                h -= h.min()
                h /= h.sum()
                H = h[:, None] * h[None, :]
                mu = convolve2d(img, H, mode='same', boundary='symm')
                img = (img - mu) / np.sqrt(convolve2d(img ** 2, H, mode='same', boundary='symm') - mu ** 2)
                img = (img - img.min())/(img.max()-img.min())
                img = np.log(img + 1)
                preprocess = 'log(... + 1) of scaled, locally contrast normalized'
            else:
                preprocess = ''
            I[..., ch2ch[int(c)]] = img.squeeze()

        I = (I - I.min()) / (I.max() - I.min())
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(I, origin='lower', interpolation='bicubic')
        ax.set_title('{} {} image'.format(preprocess, which))
        ax.axis('off')
        ax.set_aspect(1)
        figure = mpld3.fig_to_html(fig)
    else:
        flash('Could not find figure for key {}'.format(str(key)))
    return render_template('figure.html', figure=figure)


@ping
@main.route(
    '/traces/<animal_id>/<session>/<scan_idx>/<slice>/<reso_version>/<channel>/<segmentation_method>/<spike_method>')
def traces(animal_id, session, scan_idx, slice, reso_version, channel, segmentation_method, spike_method):
    key = dict(
        animal_id=animal_id, slice=slice, session=session, scan_idx=scan_idx, reso_version=reso_version,
        channel=channel, segmentation_method=segmentation_method, spike_method=spike_method
    )
    figure = None
    if reso.Activity() & key:
        traces = (reso.Activity.Trace() & key).fetch('trace', limit=20)
        traces = np.vstack(traces)
        f = traces.var(ddof=1, axis=0, keepdims=True) / traces.mean(axis=0, keepdims=True)
        traces /= f
        fps = (reso.ScanInfo() & key).fetch1('fps')
        t = np.arange(traces.shape[1]) / fps
        w = int(30 * fps)
        b = traces.shape[1] // 2
        yr = np.max(traces.max(axis=1) - traces.min(axis=1))

        fig, ax = plt.subplots(figsize=(12, 12))
        for i, tr in enumerate(traces):
            ax.plot(t[b - w:b + w], i * yr + tr[b - w:b + w], '-k')
        ax.set_xlabel('time [s]')
        ax.set_yticks([])
        ax.axis('tight')
        figure = mpld3.fig_to_html(fig)
    else:
        flash('Could not find activity for key {}'.format(str(key)))
    return render_template('figure.html', figure=figure)


@main.route('/tmp/<path:filename>')
def tmpfile(filename):
    return send_from_directory('/tmp/', filename)


@main.route('/schema/<schema>/<table>', defaults={'subtable': None}, methods=['GET', 'POST'])
@main.route('/schema/', defaults={'schema': 'experiment', 'table': 'Scan', 'subtable': None}, methods=['GET', 'POST'])
@main.route('/schema/<schema>/<table>/<subtable>', methods=['GET', 'POST'])
def relation(schema, table, subtable):
    form = RestrictionForm(request.form)
    restriction = {}
    if request.method == 'POST' and form.validate():
        restriction = form['restriction'].data

    node_attr = dict(style='filled',
                     shape='note',
                     align='left',
                     ranksep='0.1',
                     fontsize='10',
                     fontfamily='opensans',
                     height='0.2',
                     fontname='Sans-Serif'
                     )
    server = current_app.config['SERVERNAME']
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12", rankdir='LR', splines='ortho'), engine='dot')
    dot.format = 'svg'

    f = partial(lookup_class_name, context=schemata.__dict__)
    if subtable is not None:
        rel = getattr(getattr(getattr(schemata, schema), table), subtable)
    else:
        rel = getattr(getattr(schemata, schema), table)
    conn = rel.connection
    conn.dependencies.load()
    root = rel().full_table_name
    root_node = f(root) or root

    node_props = {  # http://matplotlib.org/examples/color/named_colors.html
        None: dict(fillcolor="azure4"),
        dj.Manual: dict(fillcolor='green3'),
        dj.Lookup: dict(fillcolor='azure3'),
        dj.Computed: dict(fillcolor='coral1'),
        dj.Imported: dict(fillcolor='cornflowerblue'),
        dj.Part: dict(fillcolor='azure3', fontsize='8'),
    }

    def add_node(v):

        tier = _get_tier(v)
        tmp = f(v) or v
        sc, *_ = tmp.split('.')
        with dot.subgraph(name='cluster_' + sc,
                          node_attr=node_attr,
                          graph_attr=dict(color='grey80', style='filled', label=sc)) as c:
            if tmp is not None:
                v = tmp
                kwargs = dict(zip(['schema', 'table', 'subtable'], v.split('.')))
                c.node(v, v, URL=server + url_for('main.relation', **kwargs),
                       target='_top', **node_props[tier])
            else:
                c.node(v, v, **node_props[tier])
        return v

    add_node(root)
    for node, _ in conn.dependencies.in_edges(root):
        node = add_node(node)
        dot.edge(node, root_node)

    for _, node in conn.dependencies.out_edges(root):
        node = add_node(node)
        dot.edge(root_node, node)
    filename = namehash()
    dot.render('/tmp/' + filename)

    table = djtable(rel() & restriction, limit=20)
    return render_template('schema.html', filename=filename + '.svg', table=table, form=form)
