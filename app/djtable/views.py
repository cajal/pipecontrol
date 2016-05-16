from importlib import import_module


from . import PER_PAGE
from flask.ext.login import login_required, current_user
from .forms import Restriction
from app.djtable.form_factory import DataJointFormFactory
from .relationtable import RelationTable
from flask import request, flash, url_for, render_template
from werkzeug.utils import redirect
from . import djpage
import datajoint as dj
import inspect

form_factory = DataJointFormFactory()

@djpage.route('/display/<relname>', methods=['GET', 'POST'])
@login_required
def display(relname):
    form = Restriction(request.form)
    error_msg = None

    args = dict(request.args)
    sortby = args.pop('sortby', None)
    sortdir = int(args.pop('sortdir', ['0'])[0])
    page = int(args.pop('page', ['1'])[0])
    restr = args.pop('restr', None)

    if request.method == 'POST':
        if request.form['submit'] == 'apply restriction':
            if form.validate():
                restriction = form.restriction.data.strip()
                if len(restriction) > 0:
                    if restr is not None:
                        restr.append(restriction)
                    else:
                        restr = [restriction]

    reltab = RelationTable(relname, per_page=PER_PAGE, restrictions=restr, descending=sortdir, sortby=sortby, page=page)
    return render_template('djtable/display_table.html', reltab=reltab, form=form, error_msg=error_msg)



@djpage.route('/enter/<relname>', methods=['GET', 'POST'])
#@djpage.route('/enter/<relname>/<target>', methods=['GET', 'POST'])
def enter(relname):
    enter_form = form_factory(relname)(request.form)
    if request.method == 'POST':
        if request.form['submit'] == 'Submit':
            if enter_form.validate():
                enter_form.insert()
                flash("Data has been entered in %s" % (enter_form._rel.__class__.__name__,))
                return redirect(url_for('.display', relname=relname))
            return render_template('djtable/datajoint_form.html', form=enter_form)
        return redirect(url_for('.display', relname=relname))
    else:
        for k, v in request.args.items():
            try:
                setattr(getattr(enter_form, k), 'data', v)
            except:
                pass
    return render_template('djtable/datajoint_form.html', form=enter_form,
                           target=url_for('.enter', relname=relname, target=url_for('.enter', relname=relname)))


@djpage.route('/list/<modname>')
@login_required
def list_tables(modname):
    mod = import_module(modname)
    template = '%s.{tablename}' % (modname,)

    tables = [template.format(tablename=k[0])
              for k in inspect.getmembers(mod, lambda kls: inspect.isclass(kls) and issubclass(kls,dj.BaseRelation))]
    return render_template('djtable/list_database.html', tables=tables, modname=modname)
