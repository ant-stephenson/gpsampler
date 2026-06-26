import numpy as np
from sklearn.metrics import pairwise_distances


def sample_orthogonal_matrix(n: int) -> np.ndarray:
    from scipy.stats import ortho_group

    return ortho_group.rvs(n)


def sample_with_correlation(d: int = 1) -> np.ndarray:
    U = sample_orthogonal_matrix(d)
    lam = sorted(np.random.gamma(1, 1, d), reverse=True)
    M = U @ np.diag(lam) @ U.T
    return M


def sample_rbf_kernel(
    n: int = 1, d: int = 1, sigma=1.0, ls=1.0, jitter=1e-6
) -> np.ndarray:
    if np.isscalar(ls):
        cov = np.eye(d) / (d * ls**2)
    elif len(ls) != d:
        raise ValueError(
            "Lengthscale must either be scalar or a 1-D array of dimension d."
        )
    else:
        cov = np.diag(1 / np.power(ls, 2)) / d
    X = np.random.multivariate_normal(np.zeros(d), cov, n)
    return sigma * np.exp(-pairwise_distances(X) ** 2 / 2) + np.eye(n) * jitter
