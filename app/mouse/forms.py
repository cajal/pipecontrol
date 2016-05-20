import wtforms as wtf
from commons import virus
from flask.ext.wtf import Form
from commons import mice

def mouse_validator(form, field):
    if mice.Mice() & dict(animal_id=field.data):
        return
    else:
        raise wtf.ValidationError('{0} not in database. '.format(field.data))


class SelectMouseForm(Form):
    animal_id = wtf.IntegerField('animal_id', validators=[wtf.validators.required(), mouse_validator])
    pdf = wtf.BooleanField('pdf', default=True)
    submit = wtf.SubmitField('Submit')


