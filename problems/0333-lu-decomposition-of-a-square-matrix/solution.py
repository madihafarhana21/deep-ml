import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    
    Args:
        A: Square matrix as a list of lists
    
    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    # Your code here

    U = np.array(A, dtype = float)
    n = U.shape[0]
    L = np.eye(n) # create 2-D arrays with 1s on the diagonal and zeros elsewhere

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j,i] / U[i,i]
            L[j,i] = factor
            U[j, i:] -= factor * U[i, i:]
    return L, U
    pass