import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    # Your code here

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)

    r = b - A @ x 
    p = r.copy()
    rs_old = r @ r

    for _ in range(n):
        if np.sqrt(rs_old) < tol:
            break
        Ap = A @ p 
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p 
        r = r - alpha * Ap 
        rs_new = r @ r 
        if np.sqrt(rs_new) < tol:
            break 
        p = r + (rs_new / rs_old) * p 
        rs_old = rs_new 

    return x
    pass
