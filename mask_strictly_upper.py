import numpy as np
L = 3
B = 4
rng = np.random.default_rng()
scores = rng.random(size=(B, L, L))

def create_mask_matrix(L):
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    print(i)
    print(j)
    return j > i

def mask_strictly_upper(scores):
    mask_matrix = create_mask_matrix(L)
    print(mask_matrix)
    scores[:, mask_matrix] = -np.inf
    return scores
print(mask_strictly_upper(scores))
# print(create_mask_matrix(L))