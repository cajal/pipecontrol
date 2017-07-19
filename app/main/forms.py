from .views import session
from wtforms import Form, BooleanField, StringField, PasswordField, validators, SelectField, IntegerField, FloatField
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

class TrackingForm(Form):
    exclude = BooleanField(u"Not trackable",[validators.DataRequired()],
                                         default=False)
    relative_area_threshold = FloatField(u"Threshold for relative area",
                                         [validators.DataRequired()],
                                         default=0.01)
    ratio_threshold = FloatField(u"Threshold for major/minor radius ratio threshold",
                                 [validators.DataRequired()],
                                 default=1.5)
    error_threshold = FloatField(u"Threshold for fitting error",
                                 [validators.DataRequired()],
                                 default=0.1)
    min_countour_len = IntegerField(u"Minimal contour length",
                                    [validators.DataRequired()],
                                    default=5)
    margin = FloatField(u"Minimal side margin",
                                 [validators.DataRequired()],
                                 default=0.02)
    contrast_threshold  = FloatField(u"Minimal contrast threshold",
                                 [validators.DataRequired()],
                                 default=5.)
    speed_threshold  = FloatField(u"Maximal allowed speed threshold",
                                 [validators.DataRequired()],
                                 default=.1)
    dr_threshold  = FloatField(u"Maximal allowed relative radius change threshold",
                                 [validators.DataRequired()],
                                 default=.1)
    gaussian_blur = FloatField(u"Gaussian blurring filter size",
                                 [validators.DataRequired()],
                                 default=5)