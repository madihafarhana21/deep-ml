import numpy as np

def woodbury_solve(sigma_diag, X, B):

    sigma_diag = np.asarray(sigma_diag, dtype=float)
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)

    SiX = X / sigma_diag[:, None]
    if B.ndim == 1:
        SiB = B / sigma_diag
    else:
        SiB = B / sigma_diag[:, None]

    
    D = X.shape[1]
    small = np.eye(D) + X.T @ SiX
    rhs = X.T @ SiB
    sol = np.linalg.solve(small, rhs)

    return SiB - SiX @ sol
    pass

def sherman_morrison(A_inv, u, v):

    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    Au = A_inv @ u 
    vA = v @ A_inv 
    denom = 1 + v @ Au 

    if denom == 0:
        raise ValueError("...")
    
    return A_inv - np.outer(Au, vA) / denom
    pass

def logdet_rank_one_update(logdet_A, A_inv, u, v):

    A_inv = np.asarray(A_inv, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    factor = 1 + v @ A_inv @ u 
    if factor <= 0:
        raise ValueError("...")
    
    return float(logdet_A + np.log(factor))
    pass
