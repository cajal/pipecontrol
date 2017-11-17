from collections import OrderedDict
from flask import render_template, redirect, url_for, flash, request, session, send_from_directory
import datajoint as dj
import functools
import uuid
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import graphviz
import json

from . import app, schemata, forms, tables
from .schemata import experiment, shared, reso, meso, pupil, behavior



def ping(f):
    """ Decorator to keep database connection alive."""
    def wrapper(*args, **kwargs):
        dj.conn()
        return f(*args, **kwargs)
    return wrapper

def escape_json(json_string):
    """ Clean JSON strings so they can be used as html attributes."""
    return json_string.replace('"', '&quot;')



@ping
@app.route('/')
def index():
    if not 'user' in session:
        return redirect(url_for('user'))

    return render_template('index.html')


@ping
@app.route('/user', methods=['GET', 'POST'])
def user():
    form = forms.UserForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('User switched to {}'.format(form.user.data))
        session['user'] = form.user.data
    elif 'user' in session:
        form.user.data = session['user']

    return render_template('user.html', form=form)


@ping
@app.route('/autoprocessing', methods=['GET', 'POST'])
def autoprocessing():
    form = forms.AutoProcessing(request.form)
    if request.method == 'POST' and form.validate():
        tuple_ = {'animal_id': form['animal_id'].data, 'session': form['session'].data,
                  'scan_idx': form['scan_idx'].data, 'priority': form['priority'].data}
        if not experiment.AutoProcessing().proj() & tuple_:
            experiment.AutoProcessing().insert1(tuple_)
        flash('{} inserted in AutoProcessing'.format(tuple_))

    return render_template('autoprocessing.html', form=form)


@ping
@app.route('/correction', methods=['GET', 'POST'])
def correction():
    modules = OrderedDict([('reso', reso), ('meso', meso)])

    if request.method == 'POST':
        keys = [json.loads(k) for k in request.form.getlist('channel') if k]
        module = modules[request.form['module_name']]
        module.CorrectionChannel().insert(keys, ignore_extra_fields=True)
        flash('{} key(s) inserted in CorrectionChannel'.format(len(keys)))

    all_tables = []
    user_sessions = experiment.Session() & {'username': session['user']}
    for module_name, module in modules.items():
        field_rel = ((module.ScanInfo() * module.ScanInfo.Field().proj() & user_sessions) -
                     module.CorrectionChannel())
        items = field_rel.proj('nchannels').fetch(as_dict=True)
        for item in items:
            channels = list(range(1, item['nchannels'] + 1))
            values = [escape_json(json.dumps({**item, 'channel': c})) for c in channels]
            item['channel'] = {'name': 'channel', 'options': channels, 'values': values}
        all_tables.append((module_name, tables.CorrectionTable(items)))

    return render_template('correction.html', correction_tables=all_tables)


@ping
@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    modules = OrderedDict([('reso', reso), ('meso', meso)])

    if request.method == 'POST':
        module = modules[request.form['module_name']]

        keys = [json.loads(k) for k in request.form.getlist('compartment') if k]
        keys = [{**key, 'segmentation_method': 3} for key in keys]
        module.SegmentationTask().insert(keys, ignore_extra_fields=True)
        flash('{} key(s) inserted in SegmentationTask'.format(len(keys)))

        keys = [json.loads(k) for k in request.form.getlist('ignore_item')]
        module.DoNotSegment().insert(keys, ignore_extra_fields=True)
        flash('{} key(s) ignored'.format(len(keys)))

    all_tables = []
    user_sessions = experiment.Session() & {'username': session['user']}
    compartments = experiment.Compartment().fetch('compartment')
    for module_name, module in modules.items():
        segtask_rel = ((module.ScanInfo() * shared.Channel() * module.MotionCorrection() &
                        user_sessions & 'channel <= nchannels') - module.SegmentationTask() -
                        module.DoNotSegment())
        items = segtask_rel.proj().fetch(as_dict=True)
        for item in items:
            values = [escape_json(json.dumps({**item, 'compartment': c})) for c in compartments]
            item['ignore'] = {'name': 'ignore_item', 'value': escape_json(json.dumps(item))}
            item['compartment'] = {'name': 'compartment', 'options': compartments, 'values': values}
        all_tables.append((module_name, tables.SegmentationTable(items)))

    return render_template('segmentationtask.html', segmentation_tables=all_tables)


@ping
@app.route('/progress', methods=['GET', 'POST'])
def progress():
    all_tables = []
    user_sessions = experiment.Session() & {'username': session['user']}
    for module_name, module in [('reso', reso), ('meso', meso)]:
        items = []
        for rel_name, possible_rel in module.__dict__.items():
            try:
                remaining, total = possible_rel().progress(user_sessions, display=False)
                items.append({'table': rel_name, 'processed': '{}/{}'.format(total - remaining, total),
                              'percentage': '{:.1f}%'.format(100 * (1 - remaining / total))})
            except Exception: # not a dj.Computed class
                pass
        all_tables.append((module_name, tables.ProgressTable(items)))

    return render_template('progress.html', progress_tables=all_tables)


@ping
@app.route('/jobs', methods=['GET', 'POST'])
def jobs():
    modules = OrderedDict([('reso', reso), ('meso', meso), ('behavior', behavior),
                           ('pupil', pupil)])

    if request.method == 'POST':
        to_delete = [{'key_hash': kh} for kh in request.form.getlist('delete_item')]
        jobs_rel = modules[request.form['module_name']].schema.jobs & to_delete
        num_jobs_to_delete = len(jobs_rel)
        jobs_rel.delete()
        flash('{} job(s) deleted.'.format(num_jobs_to_delete))

    all_tables = []
    fetch_attributes = ['table_name', 'status', 'key', 'user', 'error_message',
                        'connection_id', 'timestamp', 'key_hash']
    for name, module in modules.items():
        items = module.schema.jobs.proj(*fetch_attributes).fetch(as_dict=True)
        for item in items:
            item['delete'] = {'name': 'delete_item', 'value': item['key_hash']}
        all_tables.append((name, tables.JobTable(items)))

    return render_template('jobs.html', job_tables=all_tables)


@ping
@app.route('/summary', methods=['GET', 'POST'])
def summary():
    form = forms.RestrictionForm(request.form)

    summary_rel = ((reso.ScanInfo.Field() & reso.SummaryImages()).proj() +
                   (meso.ScanInfo.Field() & meso.SummaryImages()).proj())
    if request.method == 'POST' and form.validate():
        summary_rel = summary_rel & form['restriction'].data

    items = summary_rel.fetch(as_dict=True, limit=20)
    table = tables.SummaryTable(items)

    return render_template('summary.html', form=form, table=table)


@ping
@app.route('/figure/<animal_id>/<session>/<scan_idx>/<field>/<pipe_version>/<which>')
def figure(animal_id, session, scan_idx, field, pipe_version, which):
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'pipe_version': pipe_version}
    source = reso if reso.SummaryImages() & key else meso if meso.SummaryImages() & key else None

    if source is not None:
        summary_rel = source.SummaryImages.Average() * source.SummaryImages.Correlation() & key
        images, channels = summary_rel.fetch('{}_image'.format(which), 'channel')

        composite = np.zeros([*images[0].shape, 3])
        for image, channel in zip(images, channels):
            composite[..., 2 - channel] = image
        composite = (composite - composite.min()) / (composite.max() - composite.min())

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(composite, origin='lower', interpolation='lanczos')
        ax.set_title('{} image'.format(which.capitalize()))
        ax.axis('off')
        figure = mpld3.fig_to_html(fig)
    else:
        figure = None
        flash('Could not find images for {}'.format(key))

    return render_template('figure.html', figure=figure)


@ping
@app.route('/traces/<animal_id>/<session>/<scan_idx>/<field>/<pipe_version>/'
           '<segmentation_method>/<spike_method>')
def traces(animal_id, session, scan_idx, field, pipe_version, segmentation_method,
           spike_method):
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'pipe_version': pipe_version, 'channel': request.args['channel'],
           'segmentation_method': segmentation_method, 'spike_method': spike_method}
    source = reso if reso.Activity() & key else meso if meso.Activity() & key else None

    if source is not None:
        traces = np.stack((source.Activity.Trace() & key).fetch('trace', limit=20))
        f = traces.var(ddof=1, axis=0, keepdims=True) / traces.mean(axis=0, keepdims=True)
        traces /= f

        fps = (source.ScanInfo() & key).fetch1('fps')
        middle_point = traces.shape[-1] / 2
        traces = traces[:, max(0, int(middle_point - 30 * fps)): int(middle_point + 30 * fps)]
        x_axis = np.arange(traces.shape[-1]) / fps
        box_height = np.max(traces.max(axis=1) - traces.min(axis=1))

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title('Deconvolved activity for 20 cells during one minute')
        for i, trace in enumerate(traces):
            ax.plot(x_axis, i * box_height + trace, '-k')
        ax.set_xlabel('Time (secs)')
        ax.set_yticks([])
        ax.axis('tight')
        figure = mpld3.fig_to_html(fig)
    else:
        figure = None
        flash('Could not find activity traces for {}'.format(key))

    return render_template('figure.html', figure=figure)


@app.route('/tmp/<path:filename>')
def tmpfile(filename):
    return send_from_directory('/tmp/', filename)


@ping
@app.route('/schema/<schema>/<table>', defaults={'subtable': None}, methods=['GET', 'POST'])
@app.route('/schema/', defaults={'schema': 'experiment', 'table': 'Scan', 'subtable': None}, methods=['GET', 'POST'])
@app.route('/schema/<schema>/<table>/<subtable>', methods=['GET', 'POST'])
def relation(schema, table, subtable):
    f = functools.partial(dj.base_relation.lookup_class_name, context=schemata.__dict__)
    if subtable is not None:
        rel = getattr(getattr(getattr(schemata, schema), table), subtable)
    else:
        rel = getattr(getattr(schemata, schema), table)
    conn = rel.connection
    conn.dependencies.load()
    root = rel().full_table_name
    root_node = f(root) or root

    node_attr = {'style': 'filled', 'shape': 'note', 'align': 'left', 'ranksep': '0.1',
                 'fontsize': '10', 'fontfamily': 'opensans', 'height': '0.2',
                 'fontname': 'Sans-Serif'}
    node_props = {None: {'fillcolor': 'azure4'}, dj.Manual: {'fillcolor': 'green3'},
                  dj.Lookup: {'fillcolor': 'azure3'}, dj.Computed: {'fillcolor': 'coral1'},
                  dj.Imported: {'fillcolor': 'cornflowerblue'},
                  dj.Part: {'fillcolor': 'azure3', 'fontsize': '8'}}
    dot = graphviz.Digraph(node_attr=node_attr, graph_attr={'size': '12, 12', 'rankdir': 'LR',
                                                   'splines': 'ortho'}, engine='dot')
    dot.format = 'svg'
    def add_node(v):
        tier = dj.erd._get_tier(v)
        tmp = f(v) or v
        sc, *_ = tmp.split('.')
        with dot.subgraph(name='cluster_{}'.format(sc), node_attr=node_attr,
                          graph_attr={'color': 'grey80', 'style': 'filled', 'label': sc}) as c:
            if tmp is not None:
                v = tmp
                kwargs = dict(zip(['schema', 'table', 'subtable'], v.split('.')))
                c.node(v, v, URL=url_for('relation', **kwargs),
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
    filename = uuid.uuid4()
    dot.render('/tmp/{}'.format(filename))

    form = forms.RestrictionForm(request.form)
    if request.method == 'POST' and form.validate():
        rel = rel() & form['restriction'].data
    table = tables.djtable(rel(), limit=20)

    return render_template('schema.html', filename='{}.svg'.format(filename), table=table,
                           form=form)



@ping
@app.route('/tracking/<animal_id>/<session>/<scan_idx>', methods=['GET', 'POST'])
def tracking(animal_id, session, scan_idx):
    form = forms.TrackingForm(request.form)

    if request.method == 'POST' and form.validate():
        pass

    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx}
    if pupil.Eye() & key:
        prev = (pupil.Eye() & key).fetch1('preview_frames')

        fig, ax = plt.subplots(4,4,figsize=(10, 8), sharex="col", sharey="row")
        for a, fr in zip(ax.ravel(), prev.transpose([2,0,1])):
            a.imshow(fr, cmap='gray', interpolation='bicubic')
            a.axis('off')
            a.set_aspect(1)
        mpld3.plugins.connect(fig, mpld3.plugins.LinkedBrush([])) # TODO Edgar change that here
        figure = mpld3.fig_to_html(fig)
    else:
        figure = None
        flash('Could not find pupil traces for key {}'.format(str(key)))

    return render_template('trackingtask.html', form=form, figure=figure)