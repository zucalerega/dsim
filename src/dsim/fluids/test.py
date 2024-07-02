import numpy as np
import scipy
import cython
import timeit

def big_mult(n):
    for i in range(n):
        for i in range(n):
            n * n
print(timeit.timeit(lambda: big_mult(10000), number=1))
