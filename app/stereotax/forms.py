import wtforms as wtf
from fabee import lgn
import numpy as np

areas = np.unique(lgn.AtlasStereotacticTargets().fetch['area'])
selected = np.zeros_like(areas)

class StereoTacticCoordinate(wtf.Form):
    caudal = wtf.FloatField('caudal', validators=[wtf.validators.required()])
    lateral = wtf.FloatField('lateral', validators=[wtf.validators.required()])
    ventral = wtf.FloatField('ventral', validators=[wtf.validators.required()])



class StereoTacticMeasurement(wtf.Form):
    l = wtf.FormField(StereoTacticCoordinate, label='bregma')
    b = wtf.FormField(StereoTacticCoordinate, label='lambda')
    area = wtf.SelectField('area', validators=[wtf.validators.required()],
                           choices=list(zip(areas,areas)))
