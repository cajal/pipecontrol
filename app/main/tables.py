# import things
from flask_table import Table, Col

class ChannelCol(Col):
    def td_format(self, content):
        n = content[0]
        ret = "<select name='{}'>".format(content[1])
        for i in range(1,n):
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

class CorrectionChannel(Table):
    classes = ['Relation']
    animal_id = Col('animal ID')
    session = Col('Session')
    scan_idx = Col('Scan')
    reso_version = Col('Reso Version')
    slice = Col('Slice')
    channel = ChannelCol('Channel')
    select = SelectCol('Insert')

class ProgressTable(Table):
    classes = ['Relation']
    relation = Col('Relation')
    finished = Col('Finished')
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