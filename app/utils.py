import datajoint as dj
import random
import string

def ping(f):
    def ret(*args, **kwargs):
        dj.conn()
        return f(*args, **kwargs)
    return ret

def namehash(N=20):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))