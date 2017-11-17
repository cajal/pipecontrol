import flask_table


class SelectCol(flask_table.Col):
    def td_format(self, content):
        html = '<select name="{}">'.format(content['name'])
        html += '<option></option>'
        for option, value in zip(content['options'], content.get('values',
                                                                 content['options'])):
            html += '<option value="{}"{}>{}</option>'.format(value, ' selected' if
                           option == content.get('default', None) else '', option)
        html += "</select>"
        return html


class CheckBoxCol(flask_table.Col):
    def __init__(self, *args, checked=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = checked

    def td_format(self, content):
        html = '<input type="checkbox" name="{}" value="{}"{}>'.format(content['name'],
                        content['value'], ' checked' if self.checked else '')
        return html


class KeyColumn(flask_table.Col):
    def td_format(self, content):
        key = {name:content[name][0] for name in content.dtype.names} # recarray to dict
        return '<code>{}</code>'.format(key)


class CorrectionTable(flask_table.Table):
    classes = ['Relation']
    animal_id = flask_table.Col('Animal Id')
    session = flask_table.Col('Session')
    scan_idx = flask_table.Col('Scan')
    pipe_version = flask_table.Col('Pipe Version')
    field = flask_table.Col('Field')

    channel = SelectCol('Channel')


class SegmentationTable(flask_table.Table):
    classes = ['Relation']
    animal_id = flask_table.Col('Animal Id')
    session = flask_table.Col('Session')
    scan_idx = flask_table.Col('Scan')
    pipe_version = flask_table.Col('Pipe Version')
    field = flask_table.Col('Field')
    channel = flask_table.Col('Channel')

    compartment = SelectCol('Compartment')
    ignore = CheckBoxCol('Ignore')


class ProgressTable(flask_table.Table):
    classes = ['Relation']
    table = flask_table.Col('Table')
    processed = flask_table.Col('Processed')
    percentage = flask_table.Col('Percentage')


class JobTable(flask_table.Table):
    classes = ['Relation']
    table_name = flask_table.Col('Table Name')
    status = flask_table.Col('Status')
    key = KeyColumn('Key')
    user = flask_table.Col('User')
    connection_id = flask_table.Col('Connection Id')
    error_message = flask_table.Col('Error Message')
    timestamp = flask_table.DatetimeCol('Timestamp')

    delete = CheckBoxCol('Delete')


class SummaryTable(flask_table.Table):
    classes = ['Relation']
    animal_id = flask_table.Col('Animal Id')
    session = flask_table.Col('Session')
    scan_idx = flask_table.Col('Scan Idx')
    field = flask_table.Col('Field')
    pipe_version = flask_table.Col('Pipe Version')

    kwargs = {'animal_id': 'animal_id', 'session': 'session', 'scan_idx': 'scan_idx',
              'field': 'field', 'pipe_version': 'pipe_version'}
    correlation = flask_table.LinkCol('Correlation Image', 'figure', url_kwargs=kwargs,
                                      url_kwargs_extra={'which': 'correlation'})
    average = flask_table.LinkCol('Average Image', 'figure', url_kwargs=kwargs,
                                  url_kwargs_extra={'which': 'average'})
    traces = flask_table.LinkCol('Spike Traces', 'traces', url_kwargs=kwargs,
                                 url_kwargs_extra={'channel': 1, 'segmentation_method': 3,
                                                   'spike_method': 5})


def djtable(table, **kwargs):
    tbl_cls = flask_table.create_table(table.__class__.__name__)
    for col in table.heading.non_blobs:
        tbl_cls.add_column(col, flask_table.Col(col))
    for col in table.heading.blobs:
        tbl_cls.add_column(col, flask_table.Col(col))

    content = table.proj(*table.heading.non_blobs).fetch(as_dict=True, **kwargs)
    for d in content:
        for col in table.heading.blobs:
            d[col] = '- blob -'

    tbl_cls.classes = ['Relation']
    table = tbl_cls(content)

    return table