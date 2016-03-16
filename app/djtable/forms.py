import wtforms as wtf


###########################

class Restriction(wtf.Form):
    restriction = wtf.StringField('Restriction', validators=[wtf.validators.Length(max=4096)])



