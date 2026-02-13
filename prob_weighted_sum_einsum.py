import numpy as np
from einops import einsum

# AI NOTICE:
# Used AI to generate this function generate_random_sum_to_one
# As it is not critical to the problem
def generate_random_sum_to_one(n):
    random_numbers = np.random.rand(n)
    total_sum = random_numbers.sum()
    normalized_numbers = random_numbers / total_sum

    return normalized_numbers

B = 3
N = 4
D = 5
P = np.zeros((B, N))
for i in range(B):
    random_weights = generate_random_sum_to_one(N)
    P[i] = random_weights

V = np.random.randint(1, 10, size=(B, N, D))

print(P)
print(V)
def prob_weighted_sum_einsum():
    return "B N, B N D -> B D"

print(einsum(P, V, prob_weighted_sum_einsum()))
    