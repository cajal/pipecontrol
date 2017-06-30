import random
import string

def namehash(N=20):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

