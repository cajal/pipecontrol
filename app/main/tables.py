# import things
from flask import url_for, flash
from flask_table import Table, Col, DatetimeCol


class ChannelCol(Col):
    def td_format(self, content):
        n = content[0]
        ret = "<select name='{}'>".format(content[1])
        for i in range(1, n):
            ret += "<option value='{channel}'>{channel}</option>".format(channel=i)
        ret += "<option selected='selected' value='{channel}'>{channel}</option>".format(channel=n)
        ret += "</select>"
        return ret


class ChoiceCol(Col):
    def td_format(self, content):
        name, choices, default = content
        ret = "<select name='{}'>".format(name)
        for choice in choices:
            if not choice == default:
                ret += "<option value='{choice}'>{choice}</option>".format(choice=choice)
            else:
                ret += "<option selected='selected' value='{choice}'>{choice}</option>".format(choice=choice)
        ret += "</select>"
        return ret


class SelectCol(Col):
    def __init__(self, *args, checked=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = checked


    def td_format(self, content):
        if self.checked:
            return '''<input type="checkbox" name="{}" value='1' checked="checked">'''.format(content)
        else:
            return '''<input type="checkbox" name="{}" value='1'>'''.format(content)


class CheckBoxCol(Col):
    def __init__(self, *args, checked=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = checked

    def td_format(self, content):
        if self.checked:
            return '''<input type="checkbox" name="{}" value='{}' checked="checked">'''.format(*content)
        else:
            return '''<input type="checkbox" name="{}" value='{}'>'''.format(*content)


class KeyColumn(Col):
    def td_format(self, content):
        k = {kk:content[kk][0] for kk in content.dtype.fields.keys()}
        return '''<code>{}</code>'''.format(str(k))


class ResoCorrectionTable(Table):
    classes = ['Relation']
    animal_id = Col('animal ID')
    session = Col('Session')
    scan_idx = Col('Scan')
    reso_version = Col('Reso Version')
    slice = Col('Slice')
    channel = ChannelCol('Channel')
    select = CheckBoxCol('Insert', checked=False)

class MesoCorrectionTable(Table):
    classes = ['Relation']
    animal_id = Col('animal ID')
    session = Col('Session')
    scan_idx = Col('Scan')
    meso_version = Col('Meso Version')
    field = Col('Field')
    channel = ChannelCol('Channel')
    select = CheckBoxCol('Insert', checked=False)


class ProgressTable(Table):
    classes = ['Relation']
    relation = Col('Relation')
    remaining = Col('Remaining')
    total = Col('Total')
    percent = Col('Percentage')


class SegmentationTask(Table):
    classes = ['Relation']
    animal_id = Col('animal ID')
    session = Col('Session')
    scan_idx = Col('Scan')
    reso_version = Col('Reso Version')
    slice = Col('Slice')
    channel = Col('Channel')
    compartment = ChoiceCol('Compartment')
    select = SelectCol('Insert', checked=False)
    exclude = SelectCol('Exclude', checked=False)

class LinkCol(Col):

    def __init__(self, *args, label='link', **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def td_format(self, content):
        if content is not None:
            return '''<a href="{}">{}</a>'''.format(content, self.label)
        else:
            return ''

class SummaryTable(Table):
    classes = ['Relation']

    animal_id = Col('animal_id')
    session = Col('session')
    scan_idx = Col('scanidx')
    reso_version = Col('reso_version')
    slice = Col('slice')
    correlation = LinkCol('Correlation Image')
    average = LinkCol('Log Average Image')
    trace = LinkCol('Spike Trace', label='20 trace @ one min')




class JobTable(Table):

    def __init__(self, *args, target, exlude=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.exclude = exlude

    classes = ['Relation']
    allow_sort = True

    table_name = Col('table name')
    status = Col('Status')
    error_message = Col('Error Message')
    key = KeyColumn('Key')
    timestamp = DatetimeCol('Timestamp')

    delete = CheckBoxCol('Delete', checked=False)

    def sort_url(self, col_key, reverse=False):
        if reverse:
            direction =  'desc'
        else:
            direction = 'asc'
        if self.exclude is not None and not col_key in self.exclude:
            return url_for(self.target, sort=col_key, direction=direction)
        else:
            return url_for(self.target)