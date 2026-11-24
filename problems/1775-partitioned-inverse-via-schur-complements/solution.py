import numpy as np

def partitioned_inverse(M, k):

    E, F = M[:k, :k], M[:k, k:]
    G, H = M[k:, :k], M[k:, k:]

    Hi = np.linalg.inv(H)
    S = E - F @ Hi @ G 
    Si = np.linalg.inv(S)

    top_left = Si
    top_right = -Si @ F @ Hi
    bottom_left = -Hi @ G @ Si
    bottom_right = Hi + Hi @ G @ Si @ F @ Hi

    return np.block([[top_left, top_right],
                    [bottom_left, bottom_right]])
    pass

def block_slogdet(M, k):

    E, F = M[:k, :k], M[:k, k:]
    G, H = M[k:, :k], M[k:, k:]

    S = E - F @ np.linalg.inv(H) @ G

    sign_S, logdet_S = np.linalg.slogdet(S)
    sign_H, logdet_H = np.linalg.slogdet(H)

    return sign_S * sign_H, logdet_S + logdet_H

    pass

def bordered_inverse(A_inv, b, c):
    b = np.asarray(b, dtype=float)
    u = A_inv @ b 
    s = c - b @ u

    if abs(s) < 1e-12:
        raise ValueError("....")
    
    n = len(b)
    out = np.empty((n + 1, n +1))
    out[:n, :n] = A_inv + np.outer(u, u) / s
    out[:n, n] = -u / s
    out[n, :n] = -u / s
    out[n, n] = 1 / s
    return out
    pass
