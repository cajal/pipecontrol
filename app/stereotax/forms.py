import wtforms as wtf
from fabee import inj
import numpy as np
from commons import virus
from itertools import chain

areas = np.unique(inj.AtlasStereotacticTargets().fetch['area'])

subv, constr, lot = (inj.Substance() * inj.Substance.Virus() * virus.Virus()).fetch[
    'substance_id', 'construct_id', 'virus_lot']
subd, dyes = (inj.Substance() * inj.Substance.Dye()).fetch['substance_id', 'dye_name']

substances = dict(chain(zip( ['{0}@{1}'.format(a, b) for a, b in zip(constr, lot)],subv),
                        zip(dyes, subd)))
substance_names = substances.keys()

glass = inj.PipetteGlass().fetch['item_id']



class StereoTacticCoordinate(wtf.Form):
    caudal = wtf.FloatField('caudal', validators=[wtf.validators.required()])
    lateral = wtf.FloatField('lateral', validators=[wtf.validators.required()])
    ventral = wtf.FloatField('ventral', validators=[wtf.validators.required()])


class StereoTacticMeasurement(wtf.Form):
    substances = substances

    l = wtf.FormField(StereoTacticCoordinate, label='bregma')
    b = wtf.FormField(StereoTacticCoordinate, label='lambda')
    area = wtf.SelectField('area', validators=[wtf.validators.required()],
                           choices=list(zip(areas, areas)))
    substance = wtf.SelectField('substance', validators=[wtf.validators.required()],
                                choices=list(zip(substance_names, substance_names)))
    glass = wtf.SelectField('pipette', validators=[wtf.validators.required()],
                            choices=list(zip(glass, glass)))
