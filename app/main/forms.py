from .views import session
from wtforms import Form, BooleanField, StringField, PasswordField, validators, SelectField, IntegerField
from ..schemata import experiment


class UserForm(Form):
    persons = experiment.Person().fetch('username')

    user = SelectField(u"User", [validators.DataRequired()], choices=[(f, f) for f in persons])

class AutoProcessing(Form):
    animal_id = IntegerField(u"Animal ID", [validators.DataRequired()])
    session = IntegerField(u"Session", [validators.DataRequired()])
    scan_idx = IntegerField(u"Scan IDX", [validators.DataRequired()])

class SummaryForm(Form):
    animal_id = IntegerField(u"Animal ID", [validators.DataRequired()])
    session = IntegerField(u"Session", [validators.DataRequired()])
    scan_idx = IntegerField(u"Scan IDX", [validators.DataRequired()])
    slice = IntegerField(u"Slice", [validators.DataRequired()])

class RestrictionForm(Form):
    restriction = StringField(u"Restriction", [validators.DataRequired()])
