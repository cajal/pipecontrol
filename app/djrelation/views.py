from . import PER_PAGE
from flask import render_template, request, url_for
from flask.ext.login import login_user, logout_user, login_required, \
    current_user
from .forms import Restriction
from .relationtable import RelationTable
from . import djpage



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
    return render_template('djrelation/display_table.html', reltab=reltab, form=form, error_msg=error_msg)

