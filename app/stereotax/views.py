from . import stereotax
from .forms import StereoTacticMeasurement
from flask.ext.login import login_required, current_user
from flask import request, flash, url_for, render_template
from werkzeug.utils import redirect
import numpy as np
from commons import inj


def compute_coordinates(enter_form):
    site = enter_form.site.data
    target = enter_form.target.data
    atlas_caudal, atlas_lateral, atlas_ventral, la_br_basedist = \
        (inj.AtlasStereotacticTargets() & dict(injection_site=site, target_id=target)).fetch1[ \
            'caudal', 'lateral', 'ventral', 'lambda_bregma_basedist']
    c0 = np.array([enter_form.lambd.data['caudal'] - enter_form.bregma.data['caudal'],
                   enter_form.lambd.data['lateral'] - enter_form.bregma.data['lateral'],
                   0])

    cn = np.sqrt((c0 ** 2.).sum())
    correction_factor = cn / la_br_basedist

    anterior_posterior_basevec = c0 / cn
    lateral_basevec = np.array(
        [anterior_posterior_basevec[1], anterior_posterior_basevec[0], anterior_posterior_basevec[2]])
    b0 = np.array([enter_form.bregma.data['caudal'],
                   enter_form.bregma.data['lateral'],
                   enter_form.bregma.data['ventral']])

    injection_site = b0 + correction_factor * atlas_caudal * anterior_posterior_basevec + correction_factor * atlas_lateral * lateral_basevec
    depth = correction_factor * atlas_ventral
    error = np.round(
        depth * ((enter_form.lambd.data['ventral'] - enter_form.bregma.data['ventral']) / cn),
        decimals=4)

    virus_inj = dict(
        animal_id=enter_form.animal_id.data,
        virus_id=enter_form.virus.data,
        injection_site=site,
        guidance='stereotactic',
        volume=enter_form.volume.data,
        speed=enter_form.speed.data,
    )
    inj_loc = dict(
        animal_id=enter_form.animal_id.data,
        virus_id=enter_form.virus.data,
        injection_site=site,
        target_id=target,
        lambda_bregma=cn,
        caudal=np.round(correction_factor * atlas_caudal, decimals=1),
        lateral=np.round(correction_factor * atlas_lateral, decimals=1),
        ventral=np.round(correction_factor * atlas_ventral, decimals=1),
        adjustment=np.round(correction_factor, decimals=2),
    )

    return injection_site, virus_inj, inj_loc, error, correction_factor, depth


@login_required
@stereotax.route('/', methods=['GET', 'POST'])
def stereotax():
    enter_form = StereoTacticMeasurement()

    if enter_form.validate_on_submit():
        injection_site, virus_inj, inj_loc, error, correction_factor, depth = compute_coordinates(enter_form)
        if request.form['submit'] == 'insert':
            inj.VirusInjection().insert1(virus_inj)
            flash('Inserted in VirusInjection: {}'.format(repr(virus_inj)))
            inj.InjectionLocation().insert1(inj_loc)
            flash('Inserted in InjectionLocation: {}'.format(repr(inj_loc)))
            return redirect(url_for('djtable.display', relname='commons.inj.VirusInjection'))

        return render_template('stereotax/stereotactic.html',
                               form=enter_form,
                               computed=True,
                               depth=np.round(depth, decimals=1),
                               burr_hole=dict(
                                   zip(['caudal', 'lateral', 'ventral'], np.round(injection_site, decimals=1))),
                               factor=correction_factor,
                               error=error,
                               )
    return render_template('stereotax/stereotactic.html', form=enter_form, computed=False)

