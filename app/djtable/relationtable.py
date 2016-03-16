from importlib import import_module
from math import ceil

from flask import url_for


def get_get(restr=None, sortby=None, sortdir=None, page=1):
    ret = []
    if restr is not None:
        for r in restr:
            ret.append('restr=' + r)
    if sortby is not None:
        for s in sortby:
            ret.append('sortby=' + s)
    if sortdir is not None:
        ret.append('sortdir=' + str(sortdir))

    ret.append('page=%i' % page)
    if len(ret) > 0:
        return '?' + '&'.join(ret)
    else:
        return ''


class RelationTable:
    def __init__(self, relname, per_page = 20, restrictions = None, descending=0, sortby = None, page=1):
        self.relname = relname
        self.rel =  _import_relation(relname)
        self.per_page = per_page
        self.restrictions = restrictions
        self.descending = descending
        self.sortby = sortby
        self.page=page


    @property
    def attributes(self):
        return list(self.rel.heading.attributes.keys())

    def current_rows(self, page=None):
        if page is None:
            page = self.page
        if self.sortby is not None:
            if self.descending is not None:
                order_by = [self.sortby[0] + (' ASC' if self.descending == 0 else ' DESC')]
        else:
            order_by = None
        return self.current_rel.fetch(as_dict=True,
                              order_by=order_by,
                              offset=(page-1)*self.per_page,
                              limit=self.per_page
                              )

    @property
    def current_rel(self):
        if self.restrictions is not None:
            return self.rel & ' and '.join(self.restrictions)
        else:
            return self.rel


    @property
    def total_count(self):
        return len(self.current_rel)


    @property
    def pages(self):
        return int(ceil(self.total_count / self.per_page))

    @property
    def has_prev(self):
        return self.page > 1

    @property
    def has_next(self):
        return self.page < self.pages

    def reverse_sorting(self, sortby):
        if sortby == self.sortby:
            return 1-self.descending
        else:
            return self.descending

    @property
    def current_restrictions(self):
        if self.restrictions is not None:
            for r in self.restrictions:
                yield r

    @property
    def remove_constraint_iter(self):
        if self.restrictions is not None:
            for r in self.restrictions:
                yield r, self.get_url('.display', ignore_restriction=r)

    @property
    def is_restricted(self):
        return self.restrictions is not None

    def iter_pages(self, left_edge=2, left_current=2,
                   right_current=5, right_edge=2):
        last = 0
        for num in range(1, self.pages + 1):
            if num <= left_edge or \
                    (num > self.page - left_current - 1 and num < self.page + right_current) \
                    or num > self.pages - right_edge:
                if last + 1 != num:
                    yield None
                yield num
                last = num

    def get_url(self, endpoint, sortby=None, additional_restrictions=None, descending=None, page=1, ignore_restriction=None):
        restrictions = None
        if self.restrictions is None:
            if additional_restrictions is not None:
                page = 1
                restrictions = list(additional_restrictions)
        else:
            restrictions = list(self.restrictions)
            if additional_restrictions is not None:
                page = 1
                restrictions = self.restrictions + additional_restrictions

            if ignore_restriction is not None:

                page = 1
                if len(restrictions) == 1:
                    restrictions = None
                else:
                    restrictions.remove(ignore_restriction)


        return url_for(endpoint, relname=self.relname) + \
               get_get(restrictions, sortby, descending if descending is not None else self.descending, page)


def _import_relation(relname):
    p, m = relname.strip().rsplit('.', 1)
    mod = import_module(p)
    return getattr(mod, m)()