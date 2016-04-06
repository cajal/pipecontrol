from functools import wraps

import datajoint as dj


def connection_required(func):
    '''
    If you decorate a view with this, it will make sure that there connection to the databases
    gets refreshed.

    :param func: The view function to decorate.
    :type func: function
    '''
    @wraps(func)
    def decorated_view(*args, **kwargs):
        dj.conn()
        return func(*args, **kwargs)
    return decorated_view