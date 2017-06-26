
import datajoint as dj

def ping(f):
    def ret(*args, **kwargs):
        dj.conn()
        return f(*args, **kwargs)
    return ret


