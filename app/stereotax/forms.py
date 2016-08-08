import wtforms as wtf
#from fabee import inj
import numpy as np
from commons import virus, inj
from itertools import chain



class StereoTacticCoordinate(wtf.Form):
    caudal = wtf.FloatField('caudal', validators=[wtf.validators.required()])
    lateral = wtf.FloatField('lateral', validators=[wtf.validators.required()])
    ventral = wtf.FloatField('ventral', validators=[wtf.validators.required()])


class StereoTacticMeasurement(wtf.Form):

    bregma = wtf.FormField(StereoTacticCoordinate, label='bregma')
    lambd = wtf.FormField(StereoTacticCoordinate, label='lambda')
    site = wtf.SelectField('site', validators=[wtf.validators.required()])
    target = wtf.SelectField('coordinates', validators=[wtf.validators.required()])
    virus = wtf.SelectField('virus', validators=[wtf.validators.required()])

    def __init__(self,*args, **kwargs):
        super(StereoTacticMeasurement, self).__init__(*args, **kwargs)

        self.site.choices = [2*choice for choice in zip(inj.Site().fetch['injection_site'])]
        self.target.choices = [2*target for target in zip(inj.AtlasStereotacticTargets().fetch['target_id'])]
        self.virus.choices = [(vid, '{0} (Lot {1})'.format(cid, lot)) \
                              for vid, cid, lot in zip(*virus.Virus().fetch.order_by('virus_ts DESC')['virus_id','construct_id','virus_lot'])]
