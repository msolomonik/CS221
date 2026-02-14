import numpy as np

A = np.array([[2, 1, 3], [4, 5, 6]], dtype=float)
B = np.array([[7, 8], [9, 0], [1, 2]], dtype=float)

# def get_row_sum(M, row, col):
    
def matrix_grad(A, B):
    col_sums_A = np.einsum('ij->j', A)
    row_sums_B = np.einsum('ij->i', B)
    # print(col_sums_A)
    # print(row_sums_B)
    grad_A_matrix = np.array([row_sums_B for _ in range(A.shape[0])])
    grad_B_matrix = np.array([col_sums_A for _ in range(B.shape[1])]).T
    return grad_A_matrix, grad_B_matrix
    
print(matrix_grad(A, B))