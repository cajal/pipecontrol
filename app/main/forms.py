from wtforms import Form, BooleanField, StringField, PasswordField, validators, SelectField
from ..schemata import experiment


class UserForm(Form):
    persons = experiment.Person().fetch('username')

    user = SelectField(u"User", [validators.Required()], choices=[(f, f) for f in persons])
