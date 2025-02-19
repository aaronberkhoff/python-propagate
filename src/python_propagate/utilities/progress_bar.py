import time
from functools import wraps
import numpy as np

def progress(func):
    """
    Decorator that measures and prints the execution time of the decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"\nFunction '{func.__name__}' completed in {elapsed:.2f} seconds.")
        return result
    return wrapper