import random
import string

def rand_str(length=None, min_=3, max_=7):
    if length is None:
        length = random.randint(min_, max_)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))
