# Synthetic GP data
This module includes code to generate samples from a Gaussian Process using either Random Fourier Features (RFF) or Contour Integral Quadrature (CIQ). These methods generate approximate samples whose accuracy is determined by one or more fidelity parameters. See the paper [**Provably Reliable Large-Scale Sampling from Gaussian Processes**](https://arxiv.org/abs/2211.08036) for specific requirements on these parameters to ensure that datasets generated satisfy the accuracy desired. 

## GPyTorch Settings
Note that in the function ``sample_ciq_from_x`` we manually set various GPyTorch settings to ensure that iterative methods (such as CG) run to the requested number of maximum iterations, rather than terminating early due to inbuilt settings of the software.
