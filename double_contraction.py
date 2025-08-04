import numpy as np


N_DIMS = 3
VECTOR_DATA = (4, 0, 0)

def kronecker_delta(n_dim=2):
    return np.array([
        [1 if i == j else 0 for j in range(n_dim)] for i in range(n_dim)
    ])


v = np.array(VECTOR_DATA)
M = kronecker_delta(N_DIMS)

# In this case the double contraction will find the length of the vector, cause we calculating the "affection" of the evector to itself.
# This is where the square root appears in classic vector length formula.
double_contraction = np.einsum('ij,i,j->', M, v, v)

print(
    f'The double contraction on the {N_DIMS}-dimensional space of the vector {VECTOR_DATA} equals to {double_contraction}'
)
