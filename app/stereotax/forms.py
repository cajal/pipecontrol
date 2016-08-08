from collections import OrderedDict

import wtforms as wtf
# from fabee import inj
import numpy as np
from commons import virus, inj, mice

from flask.ext.wtf import Form

def mouse_validator(form, field):
    if mice.Mice() & dict(animal_id=field.data):
        return
    else:
        raise wtf.ValidationError('{0} not in database. '.format(field.data))

def target_validator(form, field):
    if inj.AtlasStereotacticTargets() & dict(injection_site=form.site.data, target_id=field.data):
        return
    else:
        raise wtf.ValidationError('Site {0} has no target {1}!'.format(form.site.data, field.data))


class StereoTacticCoordinate(Form):
    caudal = wtf.FloatField('caudal', validators=[wtf.validators.required()])
    lateral = wtf.FloatField('lateral', validators=[wtf.validators.required()])
    ventral = wtf.FloatField('ventral', validators=[wtf.validators.required()])


class StereoTacticMeasurement(Form):
    animal_id = wtf.IntegerField('animal_id', validators=[wtf.validators.required(), mouse_validator])
    bregma = wtf.FormField(StereoTacticCoordinate, label='bregma')
    lambd = wtf.FormField(StereoTacticCoordinate, label='lambda')
    site = wtf.SelectField('site', validators=[wtf.validators.required()])
    target = wtf.SelectField('target', validators=[wtf.validators.required(), target_validator])
    virus = wtf.SelectField('virus', validators=[wtf.validators.required()], coerce=int)
    volume = wtf.FloatField('volume', validators=[wtf.validators.Optional()])
    speed = wtf.FloatField('speed', validators=[wtf.validators.Optional()])


    def __init__(self, *args, **kwargs):
            super(StereoTacticMeasurement, self).__init__(*args, **kwargs)

            self.site.choices = [2 * choice for choice in \
                                 zip((inj.Site() & inj.AtlasStereotacticTargets()).fetch['injection_site'])]
            self.target.choices = [2 * target for target in zip(inj.AtlasStereotacticTargets().fetch['target_id'])]
            self._virus_data = OrderedDict([(vid, '{0} (Lot {1})'.format(cid, lot)) \
                                     for vid, cid, lot in zip(
                    *virus.Virus().fetch.order_by('virus_ts DESC')['virus_id', 'construct_id', 'virus_lot'])])

            self.virus.choices = list(self._virus_data.items())
