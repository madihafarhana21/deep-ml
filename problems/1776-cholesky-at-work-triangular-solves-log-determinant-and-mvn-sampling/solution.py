import numpy as np

def forward_sub(L, b):
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y
    pass

def back_sub(U, y):
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i,i]
    return x
    pass

def chol_solve(A, b):
    L = np.linalg.cholesky(A)
    y = forward_sub(L, b)
    return back_sub(L.T, y)
    pass

def logdet_chol(A):
    L = np.linalg.cholesky(A)
    return float(2 * np.sum(np.log(np.diag(L))))

    pass

def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)
    return eps @ L.T + mu
    pass
