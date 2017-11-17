from .schemata import experiment
import wtforms
from wtforms import validators


class UserForm(wtforms.Form):
    usernames = [(p, p) for p in experiment.Person().fetch('username')]
    user = wtforms.SelectField('User', [validators.InputRequired()], choices=usernames)


class RestrictionForm(wtforms.Form):
    restriction = wtforms.StringField('Restriction', [validators.InputRequired()])


class AutoProcessing(wtforms.Form):
    animal_id = wtforms.IntegerField('Animal Id', [validators.InputRequired()])
    session = wtforms.IntegerField('Session', [validators.InputRequired()])
    scan_idx = wtforms.IntegerField('Scan Idx', [validators.InputRequired()])
    priority = wtforms.IntegerField('Priority [0-100]', [validators.NumberRange(-128, 127)],
                                    default=50)


class TrackingForm(wtforms.Form):
    exclude = wtforms.BooleanField('Not trackable', [validators.InputRequired()],
                                   default=False)
    relative_area_threshold = wtforms.FloatField('Threshold for relative area',
                                                 [validators.InputRequired()], default=0.01)
    ratio_threshold = wtforms.FloatField('Threshold for major/minor radius ratio threshold',
                                         [validators.InputRequired()], default=1.5)
    error_threshold = wtforms.FloatField('Threshold for fitting error',
                                         [validators.InputRequired()], default=0.1)
    min_countour_len = wtforms.IntegerField('Minimal contour length',
                                            [validators.InputRequired()], default=5)
    margin = wtforms.FloatField('Minimal side margin', [validators.InputRequired()],
                                default=0.02)
    contrast_threshold = wtforms.FloatField('Minimal contrast threshold',
                                            [validators.InputRequired()], default=5)
    speed_threshold = wtforms.FloatField('Maximal allowed speed threshold',
                                         [validators.InputRequired()], default=0.1)
    dr_threshold = wtforms.FloatField('Maximal allowed relative radius change threshold',
                                      [validators.InputRequired()], default=0.1)
    gaussian_blur = wtforms.FloatField('Gaussian blurring filter size',
                                       [validators.InputRequired()], default=5)