from collections import OrderedDict, defaultdict
import datetime
import datajoint as dj
import wtforms as wtf
from wtforms.validators import required, optional
from wtforms.widgets import HiddenInput
from .relationtable import _import_relation
import re


class DataJointFormFactory:
    def __init__(self):
        self.store = {}

    def register(self, **kwargs):
        for key, rel in kwargs.items():
            if key in self.store:
                return

            if isinstance(rel, dj.Computed) or \
                    isinstance(rel, dj.Imported) or \
                    isinstance(rel, dj.Part):
                raise dj.DataJointError("Data should not be entered directly in Computed, Imported, or Part tables.")

            class ReturnValue(wtf.Form):
                _rel = rel

                @classmethod
                def append_field(cls, name, field):
                    setattr(cls, name, field)
                    return cls

                def insert(self2, replace=False):
                    rel = self2._rel
                    print(rel.connection)
                    dat = {}
                    for k, v in self2._fields.items():
                        if v.data is not None and k != 'REFERRER':  # was not specified and is also not required
                            if isinstance(v.data, datetime.datetime) or isinstance(v.data, datetime.date):
                                dat[k] = str(v.data)
                            else:
                                dat[k] = v.data
                    rel.insert1(dat, replace=replace)

            ReturnValue.required = OrderedDict()
            for name, attr in rel.heading.attributes.items():
                ReturnValue.append_field(name, field_factory(attr))
                ReturnValue.required[name] = not attr.nullable and attr.default is None
            ReturnValue.append_field('REFERRER', wtf.StringField(label='REFERRER', widget=HiddenInput()))
            self.store[key] = ReturnValue

    def __call__(self, name):
        if name not in self.store:
            rel = _import_relation(name)
            self.register(**{name: rel})
        return self.store[name]


def field_factory(attr):
    kwargs = defaultdict(list)
    kwargs['id'] = attr.name
    kwargs['label'] = attr.comment
    if attr.in_key: kwargs['validators'].append(required())
    if attr.nullable or attr.default is not None:
        kwargs['validators'].append(optional())

    if re.match('.*int.*(\(.+\))?', attr.type):
        return wtf.IntegerField(**kwargs)
    elif attr.type == 'double' or attr.type == 'float':
        return wtf.FloatField(**kwargs)
    elif attr.type.startswith('varchar'):
        ml = int(attr.type.split('(')[-1][:-1])
        return wtf.StringField(**kwargs)  # TODO: can I specify a max length here?
    elif attr.type == 'date':
        kwargs['validators'].append(date_validator)
        return wtf.DateField(format='%Y-%m-%d', default=datetime.date.today(), **kwargs)
    elif attr.type.startswith('enum'):
        choices = [(e[1:-1], e[1:-1]) for e in attr.type[attr.type.find('(') + 1:attr.type.rfind(')')].split(',')]
        return wtf.SelectField(choices=choices, **kwargs)
    elif attr.type.startswith('char'):
        l = int(attr.type.split('(')[-1][:-1])
        kwargs['validators'].append(len_validator_factory(l))
        return wtf.StringField(**kwargs)  # TODO: can I specify a max length here?
    elif attr.type == 'timestamp':
        return wtf.DateTimeField(format='%Y-%m-%d %H:%M', default=datetime.datetime.today(), **kwargs)
    else:
        raise NotImplementedError('FieldFactory does not know what to do with %s' % (attr.type))


def date_validator(form, field):
    try:
        datetime.datetime.strptime(str(field.data), '%Y-%m-%d')
    except ValueError:
        raise wtf.ValidationError("Incorrect data format, should be YYYY-MM-DD")


def len_validator_factory(n):
    def len_validator(form, field):
        if len(form.data) == n:
            return
        else:
            raise wtf.ValidationError('%s must have length %i'(form.id, n))
