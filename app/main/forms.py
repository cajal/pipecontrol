from flask.ext.wtf import Form
from wtforms import StringField, TextAreaField, BooleanField, SelectField, \
    SubmitField, SelectMultipleField, PasswordField
from wtforms.validators import required, Length, Email, Regexp
from wtforms import ValidationError
from flask.ext.pagedown.fields import PageDownField
from wtforms.widgets import TextArea, PasswordInput
from ..models import User, Role


class EditProfileForm(Form):
    name = StringField('Real name', validators=[Length(0, 64)])
    dj_user = StringField('Database Server User:', validators=[Length(0, 32)])
    dj_pass = PasswordField('DataBase Server Password', widget=PasswordInput(hide_value=False), validators=[Length(0, 32)])
    submit = SubmitField('Submit')



class EditProfileAdminForm(Form):
    email = StringField('Email', validators=[required(), Length(1, 64),
                                             Email()])
    username = StringField('Username', validators=[
        required(), Length(1, 64), Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
                                          'Usernames must have only letters, '
                                          'numbers, dots or underscores')])
    confirmed = BooleanField('Confirmed')
    name = StringField('Real name', validators=[Length(0, 64)])
    role = SelectField('Role', coerce=int)

    submit = SubmitField('Submit')

    def __init__(self, user, *args, **kwargs):
        super(EditProfileAdminForm, self).__init__(*args, **kwargs)
        self.role.choices = [(role.id, role.name)
                             for role in Role.query.order_by(Role.name).all()]
        self.user = user

    def validate_email(self, field):
        if field.data != self.user.email and \
                User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered.')

    def validate_username(self, field):
        if field.data != self.user.username and \
                User.query.filter_by(username=field.data).first():
            raise ValidationError('Username already in use.')


