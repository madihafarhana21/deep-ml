import numpy as np

def min_norm_solution(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.linalg.solve(A @ A.T, b)
    return A.T @ w
    pass

def min_norm_via_kkt(A, b):
    # return (x, lam)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    m, n = A.shape 
    K = np.block([[2*np.eye(n), A.T],
                    [A, np.zeros((m,m))]])
    rhs = np.concatenate([np.zeros(n), b])
    sol = np.linalg.solve(K, rhs)
    return sol[:n], sol[n:]
    pass

def min_norm_from_particular(A, x_p):
    A = np.asarray(A, dtype=float)
    x_p = np.asarray(x_p, dtype=float)
    return min_norm_solution(A, A @ x_p)
    pass
