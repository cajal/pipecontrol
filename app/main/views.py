from collections import OrderedDict
from inspect import isclass
from itertools import zip_longest

from flask import render_template, redirect, url_for, flash, request, session, send_from_directory, Response, send_file
import datajoint as dj
import uuid
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import graphviz
import json

from flask_weasyprint import render_pdf, HTML, CSS

from .tables import StatsTable, CellTable, create_datajoint_table
from . import main, forms, tables
from .. import schemata
from ..schemata import experiment, shared, reso, meso, stack, pupil, treadmill, tune, xcorr, mice, stimulus


def ping(f):
    """ Decorator to keep database connection alive."""

    def wrapper(*args, **kwargs):
        dj.conn().ping()
        return f(*args, **kwargs)

    return wrapper


def escape_json(json_string):
    """ Clean JSON strings so they can be used as html attributes."""
    return json_string.replace('"', '&quot;')


@ping
@main.route('/')
def index():
    if not 'user' in session:
        return redirect(url_for('main.user'))

    return render_template('index.html')


@ping
@main.route('/user', methods=['GET', 'POST'])
def user():
    form = forms.UserForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('User switched to {}'.format(form.user.data))
        session['user'] = form.user.data
    elif 'user' in session:
        form.user.data = session['user']

    return render_template('user.html', form=form)


@ping
@main.route('/autoprocessing', methods=['GET', 'POST'])
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
@main.route('/correction', methods=['GET', 'POST'])
def correction():
    modules = OrderedDict([('reso', reso), ('meso', meso), ('stack', stack)])

    if request.method == 'POST':
        keys = [json.loads(k) for k in request.form.getlist('channel') if k]
        module = modules[request.form['module_name']]
        module.CorrectionChannel().insert(keys, ignore_extra_fields=True)
        flash('{} key(s) inserted in CorrectionChannel'.format(len(keys)))

    all_tables = []
    user_sessions = experiment.Session() & {'username': session.get('user', 'unknown')}
    for module_name, module in modules.items():
        if module_name in ['reso', 'meso']:
            keys_rel = ((module.ScanInfo() * module.ScanInfo.Field().proj()
                         & user_sessions) - module.CorrectionChannel())
            correction_table = tables.CorrectionTable
        else:  # stack
            keys_rel = (module.StackInfo() & user_sessions) - module.CorrectionChannel()
            correction_table = tables.StackCorrectionTable

        items = keys_rel.proj('nchannels').fetch(as_dict=True)
        for item in items:
            channels = list(range(1, item['nchannels'] + 1))
            values = [escape_json(json.dumps({**item, 'channel': c})) for c in channels]
            item['channel'] = {'name': 'channel', 'options': channels, 'values': values}
        all_tables.append((module_name, correction_table(items)))

    return render_template('correction.html', correction_tables=all_tables)


@ping
@main.route('/segmentation', methods=['GET', 'POST'])
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
    user_sessions = experiment.Session() & {'username': session.get('user', 'unknown')}
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
@main.route('/progress', methods=['GET', 'POST'])
def progress():
    all_tables = []
    user_sessions = experiment.Session() & {'username': session.get('user', 'unknown')}
    for module_name, module in [('reso', reso), ('meso', meso), ('stack', stack)]:
        items = []
        for rel_name, possible_rel in module.__dict__.items():
            try:
                remaining, total = possible_rel().progress(user_sessions, display=False)
                items.append({'table': rel_name, 'processed': '{}/{}'.format(total - remaining, total),
                              'percentage': '{:.1f}%'.format(100 * (1 - remaining / total))})
            except Exception:  # not a dj.Computed class
                pass
        all_tables.append((module_name, tables.ProgressTable(items)))

    return render_template('progress.html', progress_tables=all_tables)


@ping
@main.route('/jobs', methods=['GET', 'POST'])
def jobs():
    modules = OrderedDict([('reso', reso), ('meso', meso), ('stack', stack),
                           ('tune', tune), ('treadmill', treadmill), ('pupil', pupil)])

    if request.method == 'POST':
        to_delete = []
        for tn_plus_kh in request.form.getlist('delete_item'):
            table_name, key_hash = tn_plus_kh.split('+')
            to_delete.append({'table_name': table_name, 'key_hash': key_hash})
        jobs_rel = modules[request.form['module_name']].schema.jobs & to_delete
        num_jobs_to_delete = len(jobs_rel)
        jobs_rel.delete()
        flash('{} job(s) deleted.'.format(num_jobs_to_delete))

    all_tables = []
    fetch_attributes = ['table_name', 'status', 'key', 'user', 'key_hash',
                        'error_message', 'timestamp']
    for name, module in modules.items():
        items = module.schema.jobs.proj(*fetch_attributes).fetch(order_by='table_name, '
                                                                          'timestamp DESC',
                                                                 as_dict=True)
        for item in items:
            value = '{}+{}'.format(item['table_name'], item['key_hash'])  # + is separator
            item['delete'] = {'name': 'delete_item', 'value': value}
            item['key_hash'] = item['key_hash'][:8] + '...'  # shorten it for display
        all_tables.append((name, tables.JobTable(items)))

    return render_template('jobs.html', job_tables=all_tables)


@ping
@main.route('/summary', methods=['GET', 'POST'])
def summary():
    form = forms.RestrictionForm(request.form)

    summary_rel = ((reso.ScanInfo.Field() & reso.SummaryImages()).proj() +
                   (meso.ScanInfo.Field() & meso.SummaryImages()).proj())
    if request.method == 'POST' and form.validate():
        summary_rel = summary_rel & form['restriction'].data

    items = summary_rel.fetch(as_dict=True, limit=25)
    table = tables.SummaryTable(items)

    return render_template('summary.html', form=form, table=table)


@ping
@main.route('/quality/', methods=['GET', 'POST'])
def quality():
    form = forms.QualityForm(request.form)

    if request.method == 'POST' and form.validate():
        key = {'animal_id': form['animal_id'].data, 'session': form['session'].data,
               'scan_idx': form['scan_idx'].data}
        pipe = reso if reso.ScanInfo() & key else meso if meso.ScanInfo() & key else None

        if pipe is not None:
            oracle = (tune.OracleMap() & key).fetch(dj.key, order_by='field')
            cos2map = (tune.Cos2Map() & key).fetch(dj.key, order_by='field')
            correlation = (pipe.SummaryImages.Correlation() & key).fetch(dj.key, order_by='field')
            average = (pipe.SummaryImages.Average() & key).fetch(dj.key, order_by='field')
            quality = (pipe.Quality.Contrast() & key).fetch(dj.key, order_by='field')
            eye = (pupil.Eye() & key).fetch1(dj.key) if pupil.Eye() & key else None

            items = [{'attribute': a, 'value': v} for a, v in (pipe.ScanInfo() & key).fetch1().items()]
            info_table = tables.InfoTable(items)

            items = []
            for schema_ in [pipe, pupil, tune]:
                for cls in filter(lambda x: issubclass(x, (dj.Computed, dj.Imported)),
                                  filter(isclass, map(lambda x: getattr(schema_, x), dir(schema_)))):
                    items.append({'relation': cls.__name__, 'populated': bool(cls() & key)})
            progress_table = tables.CheckmarkTable(items)

            return render_template('quality.html', form=form, oracle=oracle, correlation=correlation,
                                   average=average, quality=quality, eye=eye, info=info_table,
                                   progress=progress_table, cos2map=cos2map)
        else:
            flash('{} is not in reso or meso'.format(key))

    return render_template('quality.html', form=form)


@ping
@main.route('/figure/<animal_id>/<session>/<scan_idx>/<field>/<pipe_version>/<which>')
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
@main.route('/traces/<animal_id>/<session>/<scan_idx>/<field>/<pipe_version>/'
            '<segmentation_method>/<spike_method>')
def traces(animal_id, session, scan_idx, field, pipe_version, segmentation_method,
           spike_method):
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'pipe_version': pipe_version, 'channel': request.args['channel'],
           'segmentation_method': segmentation_method, 'spike_method': spike_method}
    source = reso if reso.Activity() & key else meso if meso.Activity() & key else None

    if source is not None:
        traces = np.stack((source.Activity.Trace() & key).fetch('trace', limit=25))
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


@main.route('/tmp/<path:filename>')
def tmpfile(filename):
    return send_from_directory('/tmp/', filename)


@ping
@main.route('/schema/', defaults={'schema': 'experiment', 'table': 'Scan', 'subtable': None},
            methods=['GET', 'POST'])
@main.route('/schema/<schema>/<table>', defaults={'subtable': None}, methods=['GET', 'POST'])
@main.route('/schema/<schema>/<table>/<subtable>', methods=['GET', 'POST'])
def relation(schema, table, subtable):
    graph_attr = {'size': '12, 12', 'rankdir': 'LR', 'splines': 'ortho'}
    node_attr = {'style': 'filled', 'shape': 'note', 'align': 'left', 'ranksep': '0.1',
                 'fontsize': '10', 'fontfamily': 'opensans', 'height': '0.2',
                 'fontname': 'Sans-Serif'}
    dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, engine='dot',
                           format='svg')

    def add_node(name, node_attr={}):
        """ Add a node/table to the current graph (adding subgraphs if needed). """
        table_names = dict(zip(['schema', 'table', 'subtable'], name.split('.')))
        graph_attr = {'color': 'grey80', 'style': 'filled', 'label': table_names['schema']}
        with dot.subgraph(name='cluster_{}'.format(table_names['schema']), node_attr=node_attr,
                          graph_attr=graph_attr) as subgraph:
            subgraph.node(name, label=name, URL=url_for('main.relation', **table_names),
                          target='_top', **node_attr)
        return name

    def name_lookup(full_name):
        """ Look for a table's class name given its full name. """
        pretty_name = dj.base_relation.lookup_class_name(full_name, schemata.__dict__)
        return pretty_name or full_name

    root_rel = getattr(getattr(schemata, schema), table)
    root_rel = root_rel if subtable is None else getattr(root_rel, subtable)
    root_dependencies = root_rel.connection.dependencies
    root_dependencies.load()

    node_attrs = {dj.Manual: {'fillcolor': 'green3'}, dj.Computed: {'fillcolor': 'coral1'},
                  dj.Lookup: {'fillcolor': 'azure3'}, dj.Imported: {'fillcolor': 'cornflowerblue'},
                  dj.Part: {'fillcolor': 'azure3', 'fontsize': '8'}}
    root_name = root_rel().full_table_name
    root_id = add_node(name_lookup(root_name), node_attrs[dj.erd._get_tier(root_name)])
    for node_name, _ in root_dependencies.in_edges(root_name):
        if dj.erd._get_tier(node_name) is dj.erd._AliasNode:  # renamed attribute
            node_name = root_dependencies.in_edges(node_name)[0][0]
        node_id = add_node(name_lookup(node_name), node_attrs[dj.erd._get_tier(node_name)])
        dot.edge(node_id, root_id)
    for _, node_name in root_dependencies.out_edges(root_name):
        if dj.erd._get_tier(node_name) is dj.erd._AliasNode:  # renamed attribute
            node_name = root_dependencies.out_edges(node_name)[0][1]
        node_id = add_node(name_lookup(node_name), node_attrs[dj.erd._get_tier(node_name)])
        dot.edge(root_id, node_id)

    filename = uuid.uuid4()
    dot.render('/tmp/{}'.format(filename))

    form = forms.RestrictionForm(request.form)
    if request.method == 'POST' and form.validate():
        root_rel = root_rel() & form['restriction'].data
    else:
        root_rel = root_rel()
    table = tables.create_datajoint_table(root_rel, limit=25)

    return render_template('schema.html', filename='{}.svg'.format(filename), table=table,
                           form=form)


@ping
@main.route('/tracking/<animal_id>/<session>/<scan_idx>', methods=['GET', 'POST'])
def tracking(animal_id, session, scan_idx):
    form = forms.TrackingForm(request.form)

    if request.method == 'POST' and form.validate():
        # TODO: Process input
        pass

    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx}
    if pupil.Eye() & key:
        preview_frames = (pupil.Eye() & key).fetch1('preview_frames')
        fig, axes = plt.subplots(4, 4, figsize=(10, 8), sharex=True, sharey=True)
        for ax, frame in zip(axes.ravel(), preview_frames.transpose([2, 0, 1])):
            ax.imshow(frame, cmap='gray', interpolation='lanczos')
            ax.axis('off')
            ax.set_aspect(1)
        # mpld3.plugins.connect(fig, mpld3.plugins.LinkedBrush([]))
        figure = mpld3.fig_to_html(fig)
    else:
        figure = None
        flash('Could not find eye frames for {}'.format(key))

    return render_template('trackingtask.html', form=form, figure=figure)


@main.route('/report/', methods=['GET', 'POST'])
def report():
    form = forms.ReportForm(request.form)
    if request.method == 'POST' and form.validate():
        report_type = 'scan' if form.session.data and form.scan_idx.data else 'mouse'
        if form.pdf.data:
            endpoint = 'main.{}report_pdf'.format(report_type)
        else:
            endpoint = 'main.{}report'.format(report_type)
        return redirect(url_for(endpoint, animal_id=form.animal_id.data,
                                session=form.session.data,
                                scan_idx=form.scan_idx.data))
    return render_template('report.html', form=form)


@main.route('/report/scan/<int:animal_id>-<int:session>-<int:scan_idx>')
def scanreport(animal_id, session, scan_idx):
    key = dict(animal_id=animal_id, session=session, scan_idx=scan_idx)
    pipe = reso if reso.ScanInfo() & key else meso if meso.ScanInfo() & key else None

    if pipe is not None:
        oracle = (tune.OracleMap() & key).fetch(dj.key, order_by='field')
        cos2map = (tune.Cos2Map() & key).fetch(dj.key, order_by='field')
        pxori = (tune.PixelwiseOri() & key).fetch(dj.key, order_by='field')
        cellori = bool(tune.Ori() & key)
        correlation = (pipe.SummaryImages.Correlation() & key).fetch(dj.key, order_by='field')
        average = (pipe.SummaryImages.Average() & key).fetch(dj.key, order_by='field')
        eye = (pupil.Eye() & key).fetch1(dj.key) if pupil.Eye() & key else None
        eye_track = (pupil.FittedContour() & key).fetch1(dj.key) if pupil.FittedContour() & key else None
        quality = (pipe.Quality.Contrast() & key).fetch(dj.key, order_by='field')
        oracletime = (tune.MovieOracleTimeCourse() & key).fetch(dj.key, order_by='field')
        sta = bool(tune.STA() & tune.STAQual() & key)
        xsnr = bool(xcorr.XSNR() & key)
        staext = bool(tune.STAExtent() & key)

        craniatomy_notes, session_notes = (experiment.Session() & key).fetch1('craniotomy_notes', 'session_notes')

        fields, somas, depth, height, width = (pipe.ScanInfo.Field() * pipe.ScanSet()).aggr(
            pipe.ScanSet.Unit() * pipe.ScanSet.UnitInfo() * pipe.MaskClassification.Type() & key & dict(type='soma'),
            'z', 'um_height', 'um_width', somas='count(*)').fetch('field', 'somas', 'z', 'um_height', 'um_width')
        stats = StatsTable([dict(field=f, somas=s, depth=z, height=h, width=w)
                            for f, s, z, h, w in zip(fields, somas, depth, height, width)])
        stats.items.append(
            dict(field='ALL', somas=sum([d['somas'] for d in stats.items]), depth='-', height='-', width='-')
        )

        return render_template('scan_report.html', animal_id=animal_id, session=session, scan_idx=scan_idx,
                               data=list(zip_longest(correlation, average, oracle, cos2map, fillvalue=None)),
                               craniotomy_notes=craniatomy_notes.split(','),
                               session_notes=session_notes.split(','), eye=eye, eye_track=eye_track,
                               stats=stats, sta=sta, quality=quality, oracletime=oracletime, xsnr=xsnr, staext=staext,
                               pxori=pxori, cellori=cellori)
    else:
        flash('{} is not in reso or meso'.format(key))
        return redirect(url_for('main.report'))


@main.route('/report/mouse/<int:animal_id>')
def mousereport(animal_id):
    key = dict(animal_id=animal_id)
    auto = experiment.AutoProcessing() & key

    meso_scanh = mice.Mice().aggr(meso.ScanInfo() & dict(animal_id=animal_id),
                                  time="TIME_FORMAT(SEC_TO_TIME(sum(nframes / fps)),'%%Hh %%im %%Ss')",
                                  setup="'meso'")

    stim_time = [dj.U('animal_id', 'session', 'scan_idx', 'stimulus_type').aggr(
        stim * stimulus.Condition() * stimulus.Trial() & key,
        time="TIME_FORMAT(SEC_TO_TIME(sum({})),'%%Hh %%im %%Ss')".format(duration_field))
        for duration_field, stim in zip(['cut_after', 'ori_on_secs + ori_off_secs', 'duration', 'duration'],
                                        [stimulus.Clip(), stimulus.Monet(),
                                         stimulus.Monet2(), stimulus.Varma()])
    ]

    def in_auto_proc(k):
        return bool(experiment.AutoProcessing() & k)

    stim_time = create_datajoint_table(stim_time,
                                       check_funcs=dict(autoprocessing=in_auto_proc))

    reso_scanh = mice.Mice().aggr(reso.ScanInfo() & dict(animal_id=animal_id),
                                  time="TIME_FORMAT(SEC_TO_TIME(sum(nframes / fps)),'%%Hh %%im %%Ss')",
                                  setup="'reso'")
    scanh = create_datajoint_table([reso_scanh, meso_scanh])
    scans = create_datajoint_table(
        (experiment.Scan() & auto), selection=['session', 'scan_idx', 'lens', 'depth', 'site_number', 'scan_ts']
    )
    scaninfo = create_datajoint_table(
        [(pipe.ScanInfo() & auto) for pipe in [reso, meso]],
        selection=['nfields', 'fps', 'scan_idx', 'session', 'nframes', 'nchannels', 'usecs_per_line']
    )

    ori_stats = [dj.U('animal_id').aggr(tune.Ori.Cell() & key & dict(ori_type='ori'),
                                        ori_type="'orientation'", percent_above_05='100*AVG(r2>0.01)'),
                 dj.U('animal_id').aggr(tune.Ori.Cell() & key & dict(ori_type='dir'),
                                        ori_type="'direction'", percent_above_05='100*AVG(r2>0.01)')]
    ori_stats = create_datajoint_table(ori_stats)

    stats = create_datajoint_table([experiment.Scan().aggr(
        pipe.ScanSet.Unit() * pipe.ScanSet.UnitInfo() * pipe.MaskClassification.Type() & auto & dict(type='soma'),
        somas='count(*)', scan_type='"{}"'.format(pipe.__name__)) for pipe in [reso, meso]],
        selection=['scan_type', 'session', 'scan_idx', 'somas'])
    stats.items.append(dict(scan_type='', session='ALL', scan_idx='ALL', somas=sum([e['somas'] for e in stats.items])))
    return render_template('mouse_report.html', animal_id=animal_id, scans=scans,
                           scaninfo=scaninfo, stats=stats, scanh=scanh,
                           stim_time=stim_time, ori_stats=ori_stats)


@main.route('/report/scan/<int:animal_id>-<int:session>-<int:scan_idx>.pdf')
def scanreport_pdf(animal_id, session, scan_idx):
    html = scanreport(animal_id=animal_id, session=session, scan_idx=scan_idx)
    return render_pdf(HTML(string=html),
                      stylesheets=[CSS(url_for('static', filename='styles.css')),
                                   CSS(url_for('static', filename='datajoint.css'))]
                      )


@main.route('/report/mouse/<int:animal_id>.pdf')
def mousereport_pdf(animal_id):
    html = mousereport(animal_id=animal_id)
    return render_pdf(HTML(string=html),
                      stylesheets=[CSS(url_for('static', filename='styles.css')),
                                   CSS(url_for('static', filename='pdf.css'))]
                      )
