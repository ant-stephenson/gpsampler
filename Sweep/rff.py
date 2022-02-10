import numpy as np
from functools import partial


rng = np.random.default_rng()

k_true = lambda sigma, l, xp, xq : sigma*np.exp(-0.5*np.dot(xp-xq,xp-xq)/l**2) #true kernel
z = lambda omega, D, x : np.sqrt(2/D)*np.ravel(np.column_stack((np.cos(np.dot(omega, x)),np.sin(np.dot(omega, x))))) #random features
f = lambda omega, D, w, x : np.sum(w*z(omega, D, x)) #GP approximation

def estimate_rff_kernel(X, D, ks, l):
    N,d = X.shape
    cov_omega = np.eye(d)/l**2
    omega = rng.multivariate_normal(np.zeros(d), cov_omega, D//2)
    Z = np.array([z(omega, D, xx) for xx in X])*np.sqrt(ks)
    approx_cov = np.inner(Z, Z)
    return approx_cov

def generate_rff_data(data_size, xmean, xcov_diag, noise_var, kernelscale, lenscale, D):
    assert D % 2 == 0
    input_dim = xmean.shape[0]
    assert input_dim == xcov_diag.shape[0]
    
    xcov = np.diag(xcov_diag)
    x = rng.multivariate_normal(xmean, xcov, data_size)

    cov_omega = np.eye(input_dim)/lenscale**2 #covariance matrix for fourier transform of kernel
    omega = rng.multivariate_normal(np.zeros(input_dim), cov_omega, D//2) #sample from kernel spectral density
    w = rng.standard_normal(D) #sample weights

    my_f = partial(f, omega, D, w) #GP approx as function of only x
    sample = np.array([my_f(xx) for xx in x])*np.sqrt(kernelscale)
    noisy_sample = sample + rng.normal(0.0, np.sqrt(noise_var), data_size)
    return x, sample, noisy_sample

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
    np.savetxt("x.out.gz", x)
    np.savetxt("sample.out.gz", sample)
    np.savetxt("noisy_sample.out.gz", noisy_sample)
    
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("samples have been generated")
    print("peak memory usage: %s kb" % mem)