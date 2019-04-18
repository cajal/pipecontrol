from collections import OrderedDict
from inspect import isclass
from slacker import Slacker
from datetime import datetime, timedelta
import pandas as pd
import datajoint as dj
import uuid
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import graphviz
import json
import http
from flask import render_template, redirect, url_for, flash, request, session, send_from_directory
from flask_weasyprint import render_pdf, HTML, CSS
from pymysql.err import IntegrityError

from . import main, forms, tables
from .. import schemata
from ..schemata import experiment, shared, reso, meso, stack, pupil, treadmill, tune, xcorr, mice, stimulus


def escape_json(json_string):
    """ Clean JSON strings so they can be used as html attributes."""
    return json_string.replace('"', '&quot;')



@main.route('/')
def index():
    if not 'user' in session:
        return redirect(url_for('main.user'))

    return render_template('index.html')


@main.route('/user', methods=['GET', 'POST'])
def user():
    form = forms.UserForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('User switched to {}'.format(form.user.data))
        session['user'] = form.user.data
    elif 'user' in session:
        form.user.data = session['user']

    return render_template('user.html', form=form)


@main.route('/autoprocessing', methods=['GET', 'POST'])
def autoprocessing():
    form = forms.AutoProcessing(request.form)
    if request.method == 'POST' and form.validate():
        tuple_ = {'animal_id': form['animal_id'].data, 'session': form['session'].data,
                  'scan_idx': form['scan_idx'].data, 'priority': form['priority'].data,
                  'autosegment': form['autosegment'].data}
        if not experiment.AutoProcessing().proj() & tuple_:
            experiment.AutoProcessing().insert1(tuple_)
        flash('{} inserted in AutoProcessing'.format(tuple_))

    return render_template('autoprocessing.html', form=form)


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


@main.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    modules = OrderedDict([('reso', reso), ('meso', meso)])

    if request.method == 'POST':
        module = modules[request.form['module_name']]

        keys = [json.loads(k) for k in request.form.getlist('compartment') if k]
        keys = [{**key, 'segmentation_method': 6} for key in keys]
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


@main.route('/jobs', methods=['GET', 'POST'])
def jobs():
    modules = OrderedDict([('reso', reso), ('meso', meso), ('stack', stack),
                           ('tune', tune), ('treadmill', treadmill), ('pupil', pupil),
                           ('stimulus', stimulus)])

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


@main.route('/quality/', methods=['GET', 'POST'])
def quality():
    form = forms.QualityForm(request.form)

    if request.method == 'POST' and form.validate():
        key = {'animal_id': form['animal_id'].data, 'session': form['session'].data,
               'scan_idx': form['scan_idx'].data}
        pipe = reso if reso.ScanInfo() & key else meso if meso.ScanInfo() & key else None

        if pipe is not None:
            oracle_keys = (tune.OracleMap() & key).fetch('KEY', order_by='field')
            cos2map_keys = (tune.Cos2Map() & key).fetch('KEY', order_by='field')
            summary_keys = (pipe.SummaryImages.Correlation() & key).fetch('KEY', order_by='field')
            quality_keys = (pipe.Quality.Contrast() & key).fetch('KEY', order_by='field')
            eye_key = (pupil.Eye() & key).fetch1('KEY') if pupil.Eye() & key else None

            items = []
            for schema_ in [pipe, pupil, tune]:
                for cls in filter(lambda x: issubclass(x, (dj.Computed, dj.Imported)),
                                  filter(isclass, map(lambda x: getattr(schema_, x), dir(schema_)))):
                    items.append({'relation': cls.__name__, 'populated': bool(cls() & key)})
            progress_table = tables.CheckmarkTable(items)

            items = [{'attribute': a, 'value': v} for a, v in (pipe.ScanInfo() & key).fetch1().items()]
            info_table = tables.InfoTable(items)

            return render_template('quality.html', form=form, progress_table=progress_table,
                                   info_table=info_table, oracle_keys=oracle_keys,
                                   cos2map_keys=cos2map_keys, summary_keys=summary_keys,
                                   quality_keys=quality_keys, eye_key=eye_key)
        else:
            flash('{} is not in reso or meso'.format(key))

    return render_template('quality.html', form=form)


@main.route('/figure/<animal_id>/<session>/<scan_idx>/<field>/<pipe_version>/<which>')
def figure(animal_id, session, scan_idx, field, pipe_version, which):
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'pipe_version': pipe_version}
    pipe = reso if reso.SummaryImages() & key else meso if meso.SummaryImages() & key else None

    if pipe is not None:
        summary_rel = pipe.SummaryImages.Average() * pipe.SummaryImages.Correlation() & key
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


@main.route('/traces/<animal_id>/<session>/<scan_idx>/<field>/<pipe_version>/'
            '<segmentation_method>/<spike_method>')
def traces(animal_id, session, scan_idx, field, pipe_version, segmentation_method,
           spike_method):
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx,
           'field': field, 'pipe_version': pipe_version, 'channel': request.args['channel'],
           'segmentation_method': segmentation_method, 'spike_method': spike_method}
    pipe = reso if reso.Activity() & key else meso if meso.Activity() & key else None

    if pipe is not None:
        traces = np.stack((pipe.Activity.Trace() & key).fetch('trace', limit=25))
        f = traces.var(ddof=1, axis=0, keepdims=True) / traces.mean(axis=0, keepdims=True)
        traces /= f

        fps = (pipe.ScanInfo() & key).fetch1('fps')
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
        pretty_name = dj.table.lookup_class_name(full_name, schemata.__dict__)
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
            node_name = list(root_dependencies.in_edges(node_name))[0][0]
        node_id = add_node(name_lookup(node_name), node_attrs[dj.erd._get_tier(node_name)])
        dot.edge(node_id, root_id)
    for _, node_name in root_dependencies.out_edges(root_name):
        if dj.erd._get_tier(node_name) is dj.erd._AliasNode:  # renamed attribute
            node_name = list(root_dependencies.out_edges(node_name))[0][1]
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
        endpoint = 'main.{}report{}'.format(report_type, '_pdf' if form.as_pdf.data else '')
        return redirect(url_for(endpoint, animal_id=form.animal_id.data,
                                session=form.session.data, scan_idx=form.scan_idx.data))
    return render_template('report.html', form=form)


@main.route('/report/scan/<int:animal_id>-<int:session>-<int:scan_idx>')
def scanreport(animal_id, session, scan_idx):
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx}
    pipe = reso if reso.ScanInfo() & key else meso if meso.ScanInfo() & key else None

    if pipe is not None:
        pxori_keys = (tune.PixelwiseOri() & key).fetch('KEY', order_by='field')
        quality_keys = (pipe.Quality.Contrast() & key).fetch('KEY', order_by='field')
        oracletime_keys = (tune.MovieOracleTimeCourse() & key).fetch('KEY', order_by='field')
        has_ori = bool(tune.Ori() & key)
        has_xsnr = bool(xcorr.XNR() & key)
        has_sta = bool(tune.STA() & key)
        has_staqual = bool(tune.STAQual() & key)
        has_staext = bool(tune.STAExtent() & key)
        has_eye = bool(pupil.Eye() & key)
        has_eyetrack = bool(pupil.FittedContour() & key)

        image_keys = []
        channels = shared.Channel() & 'channel <= {}'.format((pipe.ScanInfo() & key).fetch1('nchannels'))
        for field_key in (pipe.ScanInfo.Field() * channels & key).fetch('KEY'):
            field_key['has_summary'] = bool(pipe.SummaryImages() & field_key)
            field_key['has_oracle'] = bool(tune.OracleMap() & field_key)
            field_key['has_cos2map'] = bool(tune.Cos2Map() * tune.CaMovie() & field_key)
            image_keys.append(field_key)
        image_keys = list(filter(lambda k: k['has_summary'] or k['has_oracle'] or k['has_cos2map'], image_keys))

        craniotomy_notes, session_notes = (experiment.Session() & key).fetch1('craniotomy_notes', 'session_notes')
        craniotomy_notes, session_notes = craniotomy_notes.strip(), session_notes.strip()

        somas = pipe.MaskClassification.Type() & {'type': 'soma'}
        scan_somas = pipe.ScanSet.Unit() * pipe.ScanSet.UnitInfo() & {**key, 'segmentation_method': 6} & somas
        somas_per_field = pipe.ScanSet().aggr(scan_somas, avg_z='ROUND(AVG(um_z))', num_somas='count(*)')
        fields, num_somas, depths = somas_per_field.fetch('field', 'num_somas', 'avg_z')
        items = [{'field': f, 'somas': s, 'depth': z} for f, s, z in zip(fields, num_somas, depths)]
        items.append({'field': 'ALL', 'somas': sum(num_somas), 'depth': '-'})
        stats_table = tables.StatsTable(items)

        has_registration_over_time = bool(stack.RegistrationOverTime() & {'animal_id': animal_id,
                                                                          'scan_session': session})
        return render_template('scan_report.html', animal_id=animal_id, session=session, scan_idx=scan_idx,
                               craniotomy_notes=craniotomy_notes, session_notes=session_notes,
                               stats_table=stats_table, has_ori=has_ori, has_xsnr=has_xsnr, has_sta=has_sta,
                               has_staqual=has_staqual, has_staext=has_staext, image_keys=image_keys,
                               has_eye=has_eye, has_eyetrack=has_eyetrack, pxori_keys=pxori_keys,
                               quality_keys=quality_keys, oracletime_keys=oracletime_keys,
                               has_registration_over_time=has_registration_over_time)
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

    stim_time = tables.create_datajoint_table(stim_time,
                                              check_funcs=dict(autoprocessing=in_auto_proc))

    reso_scanh = mice.Mice().aggr(reso.ScanInfo() & dict(animal_id=animal_id),
                                  time="TIME_FORMAT(SEC_TO_TIME(sum(nframes / fps)),'%%Hh %%im %%Ss')",
                                  setup="'reso'")
    scanh = tables.create_datajoint_table([reso_scanh, meso_scanh])
    scans = tables.create_datajoint_table(
        (experiment.Scan() & auto), selection=['session', 'scan_idx', 'lens', 'depth', 'site_number', 'scan_ts']
    )
    scaninfo = tables.create_datajoint_table(
        [(pipe.ScanInfo() & auto) for pipe in [reso, meso]],
        selection=['nfields', 'fps', 'scan_idx', 'session', 'nframes', 'nchannels', 'usecs_per_line']
    )

    # --- orientation statistics per stack
    df1 = pd.DataFrame((stack.StackSet.Match() & key).proj('munit_id', session='scan_session').fetch())
    df2 = pd.DataFrame((tune.Ori.Cell() & key).fetch())
    df = df1.merge(df2)
    idx = df.groupby(['animal_id', 'stack_session', 'stack_idx','munit_id', 'ori_type', 'stimulus_type'])['selectivity'].idxmax()
    df3 = df.ix[idx]
    gr = df3.groupby(['animal_id', 'stack_session', 'stimulus_type','ori_type'])
    df3 = gr.agg(dict(r2=lambda x: np.mean(x > 0.01)*100)).reset_index().rename(columns={'r2':'% cells above'})

    stats = tables.create_datajoint_table([experiment.Scan().aggr(
        pipe.ScanSet.Unit() * pipe.ScanSet.UnitInfo() * pipe.MaskClassification.Type() & auto & dict(type='soma'),
        somas='count(*)', scan_type='"{}"'.format(pipe.__name__)) for pipe in [reso, meso]],
        selection=['scan_type', 'session', 'scan_idx', 'somas'])
    stats.items.append(dict(scan_type='', session='ALL', scan_idx='ALL', somas=sum([e['somas'] for e in stats.items])))
    scan_movie_oracle = bool(tune.MovieOracle() & key)
    mouse_per_stack_oracle = bool(stack.StackSet() * tune.MovieOracle() & key)
    cell_matches = bool(stack.StackSet() & key)
    stack_ori = bool(stack.StackSet() * tune.Ori() & key)
    stack_rf = bool(stack.StackSet() * tune.STAQual() & key)
    kuiper = bool(tune.Kuiper() & key)
    cell_counts = tables.create_datajoint_table(
        (stack.StackSet() & key).aggr(stack.StackSet.Unit(), unique_neurons='count(*)'))

    return render_template('mouse_report.html', animal_id=animal_id, scans=scans,
                           scaninfo=scaninfo, stats=stats, scanh=scanh,
                           stim_time=stim_time,
                           scan_movie_oracle=scan_movie_oracle, mouse_per_stack_oracle=mouse_per_stack_oracle,
                           cell_matches=cell_matches, cell_counts=cell_counts,
                           stack_ori=stack_ori, stack_rf=stack_rf, kuiper=kuiper)


@main.route('/report/scan/<int:animal_id>-<int:session>-<int:scan_idx>.pdf')
def scanreport_pdf(animal_id, session, scan_idx):
    html = scanreport(animal_id=animal_id, session=session, scan_idx=scan_idx)
    stylesheets = [CSS(url_for('static', filename='styles.css')),
                   CSS(url_for('static', filename='datajoint.css'))]
    return render_pdf(HTML(string=html), stylesheets=stylesheets)


@main.route('/report/mouse/<int:animal_id>.pdf')
def mousereport_pdf(animal_id):
    html = mousereport(animal_id=animal_id)
    stylesheets = [CSS(url_for('static', filename='styles.css')),
                   CSS(url_for('static', filename='datajoint.css'))]
    return render_pdf(HTML(string=html), stylesheets=stylesheets)


@main.route('/surgery', methods=['GET', 'POST'])
def surgery():
    fexperiment = dj.create_virtual_module("csmith_testing", "csmith_testing")
    form = forms.SurgeryForm(request.form)
    if 'user' in session:
        form['user'].data = session['user']
    if request.method == 'POST' and form.validate():
        animal_id_tuple = {'animal_id': form['animal_id'].data}
        new_surgery_id = 1
        if fexperiment.Surgery.proj() & animal_id_tuple:
            new_surgery_id = 1 + (fexperiment.Surgery & animal_id_tuple).fetch('surgery_id',
                                                                               order_by='surgery_id DESC',
                                                                               limit=1)[0]

        tuple_ = {'animal_id': form['animal_id'].data, 'surgery_id': new_surgery_id,
                  'date': str(form['date'].data), 'time': str(form['time_input'].data),
                  'username': form['user'].data, 'surgery_outcome': form['outcome'].data,
                  'surgery_quality': form['surgery_quality'].data, 'surgery_type': form['surgery_type'].data,
                  'weight': form['weight'].data, 'ketoprofen': form['ketoprofen'].data,
                  'surgery_notes': form['notes'].data}
        status_tuple_ = {'animal_id': tuple_['animal_id'], 'surgery_id': tuple_['surgery_id'], 'checkup_notes': ''}


        if not fexperiment.Surgery.proj() & tuple_:
            try:
                fexperiment.Surgery.insert1(tuple_)
                fexperiment.SurgeryStatus.insert1(status_tuple_)
                flash('Inserted record for animal {}'.format(tuple_['animal_id']))
            except IntegrityError as ex:
                ex_message = "Error: Key value not allowed. More information below."
                details = str(ex.args)
                flash(ex_message)
                flash(details)
        else:
            flash('Record already exists.')
    return render_template('surgery.html', form=form)


@main.route('/surgery/status', methods=['GET', 'POST'])
def surgery_status():
    fexperiment = dj.create_virtual_module("csmith_testing", "csmith_testing")
    date_res = (datetime.today() - timedelta(days=8)).strftime("%Y-%m-%d")
    restriction = 'surgery_outcome = "Survival" and date > "{}"'.format(date_res)
    new_surgeries = []
    for status_key in (fexperiment.Surgery & restriction).fetch():
        if len(fexperiment.SurgeryStatus & status_key) > 0:
            new_surgeries.append(((fexperiment.SurgeryStatus & status_key) * fexperiment.Surgery).fetch(order_by="timestamp DESC")[0])
    table = tables.SurgeryStatusTable(new_surgeries)
    return render_template('surgery_status.html', table=table)


@main.route('/surgery/update/<animal_id>/<surgery_id>', methods=['GET', 'POST'])
def surgery_update(animal_id, surgery_id):
    key = {'animal_id': animal_id, 'surgery_id': surgery_id}
    fexperiment = dj.create_virtual_module('csmith_testing', 'csmith_testing')
    form = forms.SurgeryEditStatusForm(request.form)
    if request.method == 'POST':
        tuple_ = {'animal_id': form['animal_id'].data, 'surgery_id': form['surgery_id'].data,
                  'day_one': int(form['dayone_check'].data), 'day_two': int(form['daytwo_check'].data),
                  'day_three': int(form['daythree_check'].data),
                  'euthanized': int(form['euthanized_check'].data), 'checkup_notes': form['notes'].data}
        try:
            fexperiment.SurgeryStatus.insert1(tuple_)
            flash("Surgery status for animal {} on date {} updated.".format(animal_id, form['date_field'].data))
        except IntegrityError as ex:
            ex_message = "Error: Key value not allowed. More information below."
            details = str(ex.args)
            flash(ex_message)
            flash(details)
        return redirect(url_for('main.surgery_status'))
    if len(fexperiment.SurgeryStatus & key) > 0:
        data = ((fexperiment.SurgeryStatus & key) * fexperiment.Surgery).fetch(order_by='timestamp DESC')[0]
        return render_template('surgery_edit_status.html', form=form, animal_id=data['animal_id'], surgery_id=data['surgery_id'],
                               date=data['date'], day_one=bool(data['day_one']), day_two=bool(data['day_two']),
                               day_three=bool(data['day_three']), euthanized=bool(data['euthanized']),
                               notes=data['checkup_notes'])
    else:
        return render_template('404.html')


@main.route('/api/v1/surgery/notification', methods=['GET'])
def surgery_notification():
    fexperiment = dj.create_virtual_module('csmith_testing', 'csmith_testing')
    num_to_word = {1: 'one', 2: 'two', 3: 'three'}

    slack_notification_channel = "#slack_api_testing"
    slack_manager = "cameron.smith"
    slacktable = dj.create_virtual_module('pipeline_notification', 'pipeline_notification')
    domain, api_key = slacktable.SlackConnection.fetch1('domain', 'api_key')
    slack = Slacker(api_key, timeout=60)

    date_res = (datetime.today() - timedelta(days=4)).strftime("%Y-%m-%d")
    restriction = 'surgery_outcome = "Survival" and date > "{}"'.format(date_res)
    surgery_data = (fexperiment.Surgery & restriction).fetch()
    for entry in surgery_data:
        if 0 < (datetime.today().date() - entry['date']).days < 4:
            status = (fexperiment.SurgeryStatus & entry).fetch(order_by="timestamp DESC")[0]
            day_key = "day_" + num_to_word[(datetime.today().date() - entry['date']).days]

            edit_url = "<{}|Update Status Here>".format(url_for('main.surgery_update',
                                                                _external=True,
                                                                animal_id=entry['animal_id'],
                                                                surgery_id=entry['surgery_id']))
            if (status['euthanized'] == 0 and status[day_key] == 0):
                manager_message = "{} needs to check animal {} for {} surgery on {}. {}".format(entry['username'].title(),
                                                                                                entry['animal_id'],
                                                                                                entry['surgery_type'],
                                                                                                entry['date'],
                                                                                                edit_url)
                ch_message = "<!channel> Reminder: " + manager_message
                slack.chat.post_message("@" + slack_manager, manager_message)
                slack.chat.post_message(slack_notification_channel, ch_message)
                if len(slacktable.SlackUser & entry) > 0:
                    slackname = (slacktable.SlackUser & entry).fetch('slack_user')
                    pm_message = "Don't forget to check on animal {} today! {}".format(entry['animal_id'],
                                                                                       edit_url)
                    slack.chat.post_message("@" + slackname, pm_message, as_user=True)
    return '', http.HTTPStatus.NO_CONTENT


@main.route('/api/v1/surgery/spawn_missing_data', methods=['GET'])
def surgery_spawn_missing_data():
    fexperiment = dj.create_virtual_module('csmith_testing', 'csmith_testing')
    if len(fexperiment.Surgery - fexperiment.SurgeryStatus) > 0:
        missing_data = (fexperiment.Surgery - fexperiment.SurgeryStatus).proj().fetch()
        for entry in missing_data:
            fexperiment.SurgeryStatus.insert1(entry)
    return '', http.HTTPStatus.NO_CONTENT

