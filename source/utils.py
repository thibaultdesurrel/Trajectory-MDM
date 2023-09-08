import numpy as np
from scipy.linalg import expm

from pyriemann.utils.distance import distance_riemann
from fastdtw import fastdtw


def r_fastdtw(x, y):
    return fastdtw(x, y, dist=distance_riemann)


def dist_F(X, Y):
    return np.linalg.norm(X - Y, ord="fro")


def exp(M, v):
    p_inv_tv = np.linalg.solve(M, v)
    return M @ expm(p_inv_tv)


def multihconj(A):
    """Vectorized matrix conjugate transpose."""
    return np.conjugate(multitransp(A))


def multilogm(A, *, positive_definite=False):
    """Vectorized matrix logarithm."""
    if not positive_definite:
        return np.vectorize(scipy.linalg.logm, signature="(m,m)->(m,m)")(A)

    w, v = np.linalg.eigh(A)
    w = np.expand_dims(np.log(w), axis=-1)
    logmA = v @ (w * multihconj(v))
    return np.real(logmA)


def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))
