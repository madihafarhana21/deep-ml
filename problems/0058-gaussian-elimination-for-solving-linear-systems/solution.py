import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # forward elimination
    for i in range(n):
        p = i + np.argmax(np.abs(A[i:, i])) # pivot row
        if p != i:
            A[[i, p]] = A[[p, i]] # swap rows of A 
            b[[i, p]] = b[[p, i]] # swap entries of b 

        for j in range(i + 1, n):
            factor = A[j,i]/A[i,i]
            A[j,i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x
    return np.zeros_like(b)
