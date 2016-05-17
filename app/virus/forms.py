import wtforms as wtf
from commons import virus
from flask.ext.wtf import Form


class NewVirusForm(Form):
    def __init__(self):
        super(NewVirusForm, self).__init__()
        if not self.virus_id.data:
            self.virus_id.data = virus.Virus().project().fetch['virus_id'].max() + 1

    virus_id = wtf.IntegerField('virus_id', validators=[wtf.validators.required()])
    construct = wtf.SelectField('construct',
                                choices=list(zip(*virus.Construct().fetch['construct_id', 'construct_id'])),
                                validators=[wtf.validators.required()])
    type = wtf.SelectField('type',
                           choices=list(zip(*virus.Type().fetch['virus_type', 'virus_type'])),
                           validators=[wtf.validators.required()])

    source = wtf.SelectField('source',
                             choices=list(zip(*virus.Source().fetch['source_id', 'source_id'])),
                             validators=[wtf.validators.required()])

    lot = wtf.StringField('virus lot', validators=[wtf.validators.Optional()])
    titer = wtf.FloatField('titer',  validators=[wtf.validators.Optional()])
    notes = wtf.TextAreaField('notes')
    submit = wtf.SubmitField('Submit')

    def enter(self):
        data = dict(
            virus_id=self.virus_id.data,
            construct_id=self.construct.data,
            virus_type=self.type.data,
            source_id=self.source.data,
            virus_notes=self.notes.data
        )
        if self.titer.data is not None:
            data['virus_titer'] = self.titer.data
        if self.lot.data is not None:
            data['virus_lot'] = self.lot.data
        virus.Virus().insert1(data)
