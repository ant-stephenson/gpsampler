#%%
from gpybench import datasets as ds
from gpybench.utils import get_off_diagonal, numpify, print_mean_std, isnotebook
from gpybench.metrics import wasserstein, kl_div_1d, nll, roberts, zscore
import gpytools.maths as gm
import gpybench.plotting as gplt
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import ExitStack
import warnings

from gpsampler.samplers import contour_integral_quad

#%%
n = 2000
ls = 0.5
ks = 1.0
d = 10
nv = 1e-4
#%%
X = torch.randn((n,d))/torch.sqrt(torch.as_tensor(d))
kernel = gpytorch.kernels.RBFKernel()
kernel.lengthscale = ls
kernel = gpytorch.kernels.ScaleKernel(kernel)
kernel.outputscale = ks

K = kernel(X).add_jitter(nv)
u = torch.randn((n,1))

#%%
f0 = gm.msqrt(K.evaluate().detach().numpy()) @ u.detach().numpy()

#%%
max_J = int(np.sqrt(n)*np.log2(n))
Js = [max_J]#[2**i for i in range(max_J)]
J = 1000#int(np.log(n)*np.sqrt(n))
Q = int(np.log(n))
# K.preconditioner_override = gpsampler.ID_Preconditioner

err = np.zeros((len(Js),2))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i,precon in enumerate([0,10000]):
        for j,J in enumerate(Js):

            with ExitStack() as stack:
                checkpoint_size = stack.enter_context(
                    gpytorch.beta_features.checkpoint_kernel(1500))
                max_preconditioner_size = stack.enter_context(
                    gpytorch.settings.max_preconditioner_size(precon))
                min_preconditioning_size = stack.enter_context(
                    gpytorch.settings.min_preconditioning_size(10))
                minres_tol = stack.enter_context(gpytorch.settings.minres_tolerance(1e-10))
                solves, weights, _, _ = contour_integral_quad(
                            K,
                            u,
                            max_lanczos_iter=J, num_contour_quadrature=Q)
            f = (solves * weights).sum(0).squeeze()

            err[j,i] = np.sqrt(np.sum((f.detach().numpy()-f0)**2)/n)
# %%
print(err)