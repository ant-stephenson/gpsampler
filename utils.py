import pathlib
import re
import numpy as np
from functools import singledispatch
from typing import Tuple
from scipy.special import gamma, binom
from scipy.optimize import root, fsolve, minimize
from scipy.integrate import quad

from gpprediction.utils import k_se, k_mat_half as k_exp


@singledispatch
def check_exists(path: pathlib.Path, expt_no=None, overwrite=False,
                 suffix=".csv") -> Tuple[pathlib.Path, int]:
    if path.with_suffix(suffix).exists() and not overwrite:
        file_name = path.stem
        prev_no = re.findall("\d$", str(path))
        if prev_no:
            expt_no = int(prev_no[0]) + 1
            path = path.with_name(f"{file_name[:-len(prev_no)]}{expt_no}")
        else:
            expt_no = 1
            path = path.with_name(f"{file_name}_{expt_no}")
        return check_exists(path, expt_no, overwrite, suffix)
    else:
        return path.with_suffix(suffix), expt_no


@check_exists.register
def check_exists_str(
        path: str, expt_no=None, overwrite=False, suffix=".csv") -> Tuple[
        pathlib.Path, int]:
    _path = pathlib.Path(path)
    return check_exists(_path, expt_no, overwrite, suffix)


def am(m: int, ks=1.0, ls=1.0, d=1) -> float:
    """Expected value (over x) of RBF kernel element

    Args:
        m (int): mth moment
        b (float, optional): _description_. Defaults to 1.0.
        ls (float, optional): _description_. Defaults to 1.0.
        d (int, optional): _description_. Defaults to 1.

    Returns:
        float: _description_
    """
    return ks**m*(1+2*m/(d*ls**2))**(-d/2)


def gamma_density(nu, rho, s): return (
    nu/rho**2)**nu * s**(nu-1)/gamma(nu) * np.exp(-nu/rho**2 * s)

# Ek_nu:


def k1_integrand(d, nu, rho, s): return (
    1+2/(d*s))**(-d/2) * gamma_density(nu, rho, s)

# Ek_nu^2:


def k2_integrand(d, nu, rho, s): return (
    1+4/(d*s))**(-d/2) * gamma_density(nu, rho, s)


# Ekk_nu:
def k3_integrand(d, nu, rho, s, ls): return (
    2/d)**(-d/2) * (d/2 + 1/ls**2 + 1/s)**(-d/2) * gamma_density(nu, rho, s)


def Eknu(d, nu, rho) -> Tuple[float, float]:
    anu, anu_err = quad(lambda s: k1_integrand(d, nu, rho, s), 0, np.inf)
    return anu, anu_err


def Eknu2(d, nu, rho) -> Tuple[float, float]:
    a2nu, a2nu_err = quad(lambda s: k2_integrand(d, nu, rho, s), 0, np.inf)
    return a2nu, a2nu_err


def rescale_dataset_noise(data: np.ndarray, new_nv: float,
                          rng: np.random.Generator) -> np.ndarray:
    """ Takes in a numpy array of X-y data of shape N x d+1 where the last
    column is a vector of observations (y) and the first d columns are the input
    X data. 

    This function takes the final y vector and scales the noise variance to
    whatever is desired as the input, simultaneously rescaling the output scale
    variance (so the two sum to 1) and outputs a modified numpy array where the
    final column has been replaced with the transformed version. 

    Args:
        data (np.ndarray): Nxd+1 array of X-y data
        new_nv (float): desired noise variance
        rng (np.random.Generator): in-use random generator

    Returns:
        np.ndarray: array with same X data and transformed y data
    """
    n = data.shape[0]
    orig_nv = 0.008
    orig_ks = 1-orig_nv
    eta = (new_nv-orig_nv)/(1-orig_nv)
    xi = rng.standard_normal((n,)) * np.sqrt(np.abs(eta))
    y1 = np.sqrt(1-eta) * data[:, -1] + np.sign(eta) * xi
    data[:, -1] = y1
    return data


def a(d): return d/(4)
def b(l): return 1/(2*l**2)
def c(d, l): return np.sqrt(a(d)**2 + 2*a(d)*b(l))


def A(d, l): return a(d) + b(l) + c(d, l)
def B(d, l): return b(l)/A(d, l)


def rbf_mat_idx_to_op_idx(mat_idx: int, d: int) -> int:
    """Only really works for large enough n and mat_idx (unsurprisingly)

    Args:
        mat_idx (int): _description_
        d (int): _description_

    Returns:
        int: _description_
    """
    def fun(k): return mat_idx - binom(k+d, d)
    return int(np.ceil(fsolve(fun, 2).item())) + 1


def compute_rbf_eigval_gpml(n: int, d: int, l: float, k: int) -> float:
    """Computes the k^th eigenvalue of an RBF Gram matrix with N(0,1/dI_d)
    x-data. Uses GPML (4.41) p98 

    Args:
        n (int): Dataset size
        d (int): Input dimension
        l (float): lengthscale
        k (int): eigenvalue-index

    Returns:
        float: the eigenvalue
    """
    return n * (2*a(d)/A(d, l))**(d/2)*B(d, l)**(k-1)


def compute_rbf_eigval_UB(n: int, d: int, l: float, k: int) -> float:
    return n * (2*a(d)/A(d, l))**(d/2) * B(d, l)**(k**(1/d))


def compute_rbf_cond(n: int, d: int, l: float, nv: float) -> float:
    lambda_1 = compute_rbf_eigval_gpml(n, d, l, 1)
    return lambda_1 / nv


def compute_J_noprecond(
        n: int, nv: float, cond: float, eps=0.1, eta=0.8, C=10) -> int:
    delta = 0.9 * eps*np.sqrt(nv*(1-eta))
    J = 1 + np.sqrt(cond)/2 * (np.log(cond*np.sqrt(nv*n)
                                      ) + 2*np.log(np.log(cond)) - np.log(eps*np.sqrt(nv*(1-eta))-delta) + C)
    return J


def compute_J_precond(
        n: int, d: int, l: float, nv: float, ks: float, nu: float, eps=0.1,
        eta=0.8, C=10) -> int:
    rootn_eig = compute_sqrtnth_eigval_UB(n, d, l, nu, ks)
    delta = 0.9 * eps*np.sqrt(nv*(1-eta))
    J = 1 + np.sqrt(rootn_eig) * n ** (3/8) / (np.sqrt(eta * nv)
                                               ) * (5/4 * np.log(n) - np.log(eps*np.sqrt(nv*(1-eta))-delta) + C)
    return J


def compute_rbf_eigenvalue(n: int, d: int, l: float, k: int) -> float:
    """Computes the k^th eigenvalue of an RBF Gram matrix with N(0,1/dI_d)
    x-data. Uses GPML (4.41) p98 

    Args:
        n (int): Dataset size
        d (int): Input dimension
        l (float): lengthscale
        k (int): eigenvalue-index

    Returns:
        float: the eigenvalue
    """
    alphasq = d/2
    epssq = 1/(2*l**2)
    deltasq = alphasq/2 * (np.sqrt(1+4*epssq/alphasq)-1)
    term1 = (alphasq / (alphasq + deltasq + epssq))**d
    term2 = (1+deltasq/epssq + alphasq/epssq)**(d-k)

    return n * term1 * term2


def compute_sqrtnth_rbf_eigval(n: int, d: int, l: float) -> float:
    k = int(((np.sqrt(n)+1) * gamma(d+1))**(1/d))-1
    return compute_rbf_eigval_gpml(n, d, l, k)


def compute_exp_max_eigval(n: int, d: int, l: float) -> float:
    """Computes the k^th eigenvalue of an Exp Gram matrix with U(0,1;d)
    x-data. Uses ... 

    Args:
        n (int): Dataset size
        d (int): Input dimension
        l (float): lengthscale
        k (int): eigenvalue-index

    Returns:
        float: the eigenvalue
    """
    def trans_eqn(w): return w*l*np.tan(w.sum()/2)-1
    omega = root(trans_eqn, np.ones(d)).x
    return n * (2*l*np.sqrt(np.pi))**d * gamma((d+1)/2)/np.sqrt(np.pi) * (1 + l**2*np.dot(omega, omega))**(-(d+1)/2)


def compute_max_eigval_UB(
        n: int, d: int, l: float, nu: float, ks: float = 1.0) -> float:
    """Assumes x ~ N(0,d^-1I_d). Bound holds with probability at least 1-O(n^-1/2).

    Args:
        n (int): _description_
        d (int): _description_
        l (float): _description_
        nu (float): set to np.inf is using SE kernel

    Returns:
        float: _description_
    """
    if np.isfinite(nu):
        a, _ = Eknu(d, nu, l)
        v = Eknu2(d, nu, l)[0] - a**2
    else:
        a = am(1, ks=ks, ls=l, d=d)
        v = am(2, ks=ks, ls=l, d=d) - a**2
    return ks + (n-1) * (a + np.sqrt(v))


def compute_sqrtnth_eigval_UB(
        n: int, d: int, l: float, nu: float, ks: float = 1.0) -> float:
    """Assumes x ~ N(0,d^-1I_d). Bound holds with probability at least 1-O(n^-1/2).

    Args:
        n (int): _description_
        d (int): _description_
        l (float): _description_
        nu (float): _description_

    Returns:
        float: _description_
    """
    norm2 = compute_max_eigval_UB(n, d, l, nu)
    return ks + np.sqrt((norm2*ks - ks**2)*(np.sqrt(n)-1))


def estimate_cond(
        n: int, m: int, d: int, l: float, delta: float, nv: float, ks: float,
        kernel_type: str) -> float:
    """ using probabilistic guarantees use submatrix to estimate condition
    number and then increase by a small amount to ensure that Pr{|submat_eig -
    actual_eig| >= eps} < delta
    TODO: generalise for general GPyTorch kernels (would need to replace str
    arg)

    Args:
        n (int): _description_
        m (int): _description_
        d (int): _description_
        l (float): _description_
        delta (float): _description_
        nv (float): _description_
        ks (float): _description_
        kernel_type (str): _description_

    Returns:
        float: _description_
    """
    if kernel_type.lower() == 'rbf':
        # check theory holds for this:
        return compute_rbf_cond(n, d, l, nv)
    elif kernel_type.lower() == 'exp':
        x = np.random.randn(m, d) / np.sqrt(d)
        K = k_exp(x, x, 1, l)
        # could do better using other eigenvalues...
        submat_eig = np.linalg.eigvalsh(K)
        Rmax = (submat_eig[:-1]/(1-submat_eig[:-1]/submat_eig[-1])).mean()
        eps = Rmax + np.sqrt(-ks**2 * np.log(delta/2)/m)
        cond = (submat_eig[-1] + eps) / nv
        return cond
    else:
        raise NotImplementedError


def compute_submat_sz(eps, ks, delta):
    def optim(m): return 2*(m-1) * np.exp(-2*eps**2*m/ks**2) - delta
    mstar = fsolve(optim, 100).item()
    return int(np.ceil(mstar))
