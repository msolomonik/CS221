import numpy as np
from einops import einsum
import math

B = 4
M = 3
D = 5
N = 2

A = np.random.randint(1, 10, size=(B, M, D))
C = np.random.randint(1, 10, size=(B, N, D))

def normalized_inner_products(A, C, normalize=True):
    new_matrix = einsum(A, C, "B M D, B N D -> B M N")
    if normalize:
        # return new_matrix / math.sqrt(D)
        # above is what I originally wrote -- GPT-5.2 caught my mistake
        # am incorrectly using global constant instead of input
        return new_matrix / math.sqrt(A.shape[-1])
    return new_matrix
    # outer: B M N
    # inner: B M D, B N D

print(normalized_inner_products(A, C))