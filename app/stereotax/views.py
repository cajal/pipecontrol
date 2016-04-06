from ..djtable.decorators import connection_required
from . import stereotax
from .forms import StereoTacticMeasurement
from flask.ext.login import login_required, current_user
from flask import request, flash, url_for, render_template
from werkzeug.utils import redirect
import numpy as np
from fabee import lgn

@login_required
@stereotax.route('/', methods=['GET', 'POST'])
@connection_required
def stereotax():
    enter_form = StereoTacticMeasurement(request.form)
    if request.method == 'POST':
        if request.form['submit'] == 'Submit':
            if enter_form.validate():
                area = enter_form.area.data
                coord = (lgn.AtlasStereotacticTargets() & "area='%s'" % (area,)).fetch1()
                c0 = np.array([enter_form.l.data['caudal'] - enter_form.b.data['caudal'],
                               enter_form.l.data['lateral'] - enter_form.b.data['lateral'],
                               0])

                cn = np.sqrt((c0 ** 2.).sum())
                f = np.round(cn / coord['lambda_bregma_basedist'], decimals=2)

                c = c0 / cn
                l = np.array([c[1], -c[0], c[2]])
                b0 = np.array([enter_form.b.data['caudal'],
                               enter_form.b.data['lateral'],
                               enter_form.b.data['ventral']])

                burr_hole = b0 + f * coord['caudal'] * c + f * coord['lateral'] * l
                depth = f * coord['ventral']
                error = np.round(depth * ((enter_form.l.data['ventral'] - enter_form.b.data['ventral']) / c0[0]),
                                 decimals=4)

                insert_url = url_for('djtable.enter', relname='fabee.lgn.Injections') + \
                             "?lambda_bregma=%.2f&caudal=%.1f&lateral=%.1f&ventral=%.1f&adjustment=%.2f&area=%s" % (
                                        np.abs(c0[0]), f * coord['caudal'],
                                        f * coord['lateral'], f * coord['ventral'], f, area)

                return render_template('stereotax/stereotactic.html', form=enter_form,
                                       computed=True,
                                       depth=np.round(depth, decimals=1),
                                       burr_hole=dict(zip(['caudal', 'lateral', 'ventral'],
                                                          np.round(burr_hole, decimals=1))),
                                       factor=f,
                                       error=error,
                                       insert_url=insert_url
                                       )
            return render_template('stereotax/stereotactic.html', form=enter_form, computed=False)
        return redirect(url_for('index'))

    return render_template('stereotax/stereotactic.html', form=enter_form, computed=False)
