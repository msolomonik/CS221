import numpy as np
import einops

B = 3
D = 8
G = 4
x = np.random.randint(1, 10, size=(B, D))

print(einops.rearrange(x, "B (G M) -> B G M",G=G, M=2))
# input B X D
# output: B X G X (D // G)