import numpy as np
from functools import partial
import torch
import gpytorch
from gpytorch.utils import contour_integral_quad
from typing import Tuple, Optional


rng = np.random.default_rng()

k_true = lambda sigma, l, xp, xq : sigma*np.exp(-0.5*np.dot(xp-xq,xp-xq)/l**2) #true kernel
z = lambda omega, D, x : np.sqrt(2/D)*np.ravel(np.column_stack((np.cos(np.dot(omega, x)),np.sin(np.dot(omega, x))))) #random features
f = lambda omega, D, w, x : np.sum(w*z(omega, D, x)) #GP approximation

def estimate_rff_kernel(X: np.ndarray, D: int, ks: float, l: float) -> np.ndarray:
    N,d = X.shape
    cov_omega = np.eye(d)/l**2
    omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)
    Z = np.array([z(omega, D, xx) for xx in X])*np.sqrt(ks)
    approx_cov = np.inner(Z, Z)
    return approx_cov

def construct_kernels(l: float, b:float=1.0) -> gpytorch.kernels.Kernel:
    kernel = gpytorch.kernels.RBFKernel()
    kernel.lengthscale = l
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    kernel.outputscale = b
    return kernel

def estimate_ciq_kernel(X, J, Q, ks, l) -> np.ndarray:
    raise NotImplementedError
    kernel = construct_kernels(l, ks)
    n, d = X.shape
    J = int(np.sqrt(n) * np.log(n))
    Q = int(np.log(n))
    solves, weights, _, _ = contour_integral_quad(kernel(X).evaluate_kernel(), torch.eye(n).double(), max_lanczos_iter=J, num_contour_quadrature=Q)
    Ksqrt = (solves * weights).sum(0)

def generate_ciq_data(n: int, xmean: np.ndarray, xcov_diag: np.ndarray, noise_var: float, kernelscale: float, lenscale: float, J: int, Q: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generates a data sample from a MVN and a sample from an approximate GP
    using CIQ to approximate K^1/2 b

    Args:
        n (int): Length of sample
        xmean (np.ndarray): Mean of x distribution
        xcov_diag (np.ndarray): Variances of x values
        noise_var (float): Noise variance of GP
        kernelscale (float): scaling factor for GP kernel
        lenscale (float): RBF lengthscale
        J (int): # Lanczsos iterations
        Q (int): # Quadrature points

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: sampled x values, noise-free
   sample and noisy GP sample
    """
    input_dim = xmean.shape[0]
    assert input_dim == xcov_diag.shape[0]

    xcov = np.diag(xcov_diag)
    x = rng.multivariate_normal(xmean, xcov, n)
    u = rng.standard_normal(n)

    kernel = construct_kernels(lenscale, kernelscale)
    solves, weights, _, _ = contour_integral_quad(kernel(x).evaluate_kernel(), torch.tensor(u), max_lanczos_iter=J, num_contour_quadrature=Q)
    f = (solves * weights).sum(0).detach().numpy()
    noisy_sample = f + rng.normal(0.0, np.sqrt(noise_var), n)
    return x, sample, noisy_sample 


def generate_rff_data(n: int, xmean: np.ndarray, xcov_diag: np.ndarray, noise_var: float, kernelscale: float, lenscale: float, D: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generates a data sample from a MVN and a sample from an approximate GP using RFF

    Args:
        n (int): Length of sample
        xmean (np.ndarray): Mean of x distribution
        xcov_diag (np.ndarray): Variances of x values
        noise_var (float): Noise variance of GP
        kernelscale (float): scaling factor for GP kernel
        lenscale (float): RBF lengthscale
        D (int): # RFF

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: sampled x values, noise-free
   sample and noisy GP sample
    """
    assert D % 2 == 0
    input_dim = xmean.shape[0]
    assert input_dim == xcov_diag.shape[0]
    
    xcov = np.diag(xcov_diag)
    x = rng.multivariate_normal(xmean, xcov, n)

    cov_omega = np.eye(input_dim)/lenscale**2 #covariance matrix for fourier transform of kernel
    omega = rng.multivariate_normal(np.zeros(input_dim), cov_omega, D//2) #sample from kernel spectral density
    w = rng.standard_normal(D) #sample weights

    my_f = partial(f, omega, D, w) #GP approx as function of only x
    sample = np.array([my_f(xx) for xx in x])*np.sqrt(kernelscale)
    noisy_sample = sample + rng.normal(0.0, np.sqrt(noise_var), n)
    return x, sample, noisy_sample

def sample_rff_from_x(x: np.ndarray, sigma: float,noise_var: float,l: float, rng: np.random.Generator, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Generates sample from approximate GP using RFF method at points x

    Args:
        x (np.ndarray): Nxd matrix of locations
        sigma (float): outputscale
        noise_var (float): noise variance
        l (float): lengthscale
        rng (Generator): RNG
        D (int): Number of RFF

    Returns:
        Tuple[np.ndarray, np.ndarray]: Approx. GP draw; 1D array of length n and approx cov
    """
    n,d = x.shape
    cov_omega = np.eye(d)/l**2
    omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)

    w = rng.standard_normal(D)
    my_f = partial(f, omega, D, w)
    y = np.array([my_f(xx) for xx in x])*np.sqrt(sigma)
    noise = rng.normal(scale=np.sqrt(noise_var), size=n)
    y_noise = y + noise

    my_z = partial(z, omega, D)
    Z = np.array([my_z(xx) for xx in x])*np.sqrt(sigma)
    approx_cov = np.inner(Z,Z)
    return y_noise, approx_cov

def sample_ciq_from_x(x: np.ndarray, sigma: float,noise_var: float,l: float, rng: np.random.Generator, J: int, Q: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    """ Generates sample from approximate GP using RFF method at points x

    Args:
        x (np.ndarray): Nxd matrix of locations
        sigma (float): outputscale
        noise_var (float): noise variance
        l (float): lengthscale
        rng (Generator): RNG
        D (int): Number of RFF

    Returns:
        Tuple[np.ndarray, np.ndarray]: Approx. GP draw; 1D array of length n and approx cov
    """
    n, d = x.shape
    u = rng.standard_normal(n)

    kernel = construct_kernels(l, sigma)
    solves, weights, _, _ = contour_integral_quad(kernel(torch.tensor(x)).evaluate_kernel(), torch.tensor(u.reshape(-1,1)), max_lanczos_iter=J, num_contour_quadrature=Q)
    f = (solves * weights).sum(0).detach().numpy().squeeze()
    y_noise = f + rng.normal(0.0, np.sqrt(noise_var), n)
    approx_cov = np.nan * np.ones((n,n))
    return  y_noise, approx_cov

if __name__ == '__main__':
    N = 1000 # no. of data points
    d = 10 #input (x) dimensionality
    D = 1000 #no.of fourier features
    l = 1.1 #lengthscale
    sigma = 0.7 #kernel scale
    noise_var = 0.2 #noise variance
    
    xmean = np.zeros(d)
    xcov_diag = np.full(d, 1.0/d)
    
    print(
    """
data_size %d
xmean %s
xcov_diag %s
noise_var %.2f
kernelscale %.2f
lenscale %.2f
    """
            % (
                N, 
                str(xmean), 
                str(xcov_diag), 
                noise_var, 
                sigma, 
                l
            )
        )
    
    x, sample, noisy_sample = generate_rff_data(N, xmean, xcov_diag, noise_var, sigma, l, D)
    # np.savetxt("x.out.gz", x)
    # np.savetxt("sample.out.gz", sample)
    # np.savetxt("noisy_sample.out.gz", noisy_sample)
    
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("samples have been generated")
    print("peak memory usage: %s kb" % mem)