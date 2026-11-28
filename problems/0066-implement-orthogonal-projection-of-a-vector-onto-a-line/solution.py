
def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """

    num = sum(x * y for x, y in zip(v, L))
    denom = sum(y * y for y in L)

    return [((num/denom) * x) for x in L]

    pass
