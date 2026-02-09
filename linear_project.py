import numpy as np
from einops import einsum

B = 4
D_in = 3
D_out = 5
MIN = 1
MAX = 10

x = np.random.randint(MIN, MAX, size=(B, D_in))
W = np.random.randint(MIN, MAX, size=(D_in, D_out))
b = np.random.randint(MIN, MAX, size=(D_out,))

def linear_project(x, W, b):
    return einsum(x, W, "B D_in, D_in D_out -> B D_out") + b
print(linear_project(x, W, b))