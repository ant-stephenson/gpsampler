"""Pre-asymptotic exponent check: Matern neff under Gaussian vs uniform inputs.
Confirms the per-decade neff exponent under a (sub-exponential) Gaussian design
approaches the theoretical theta = d/(2nu+d) from above as n grows, consistent
with the same asymptotic rate as the bounded-domain (uniform) design.
"""

import numpy as np
from scipy.spatial.distance import cdist


def matern(D2, l, nu):
    r = np.sqrt(np.maximum(D2, 0)) / l
    if nu == 1.5:
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    if nu == 2.5:
        return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    raise ValueError(nu)


def neff(X, l, nu, sig2):
    K = matern(cdist(X, X, "sqeuclidean"), l, nu)
    lam = np.linalg.eigvalsh(K)
    return float(np.sum(lam / (lam + sig2)))


def local_exponents(gen, d, nu, l, ns, sig2=1e-2, seeds=(0, 1)):
    nev = [
        np.mean([neff(gen(n, d, s), l, nu, sig2) for s in seeds]) for n in ns
    ]
    loc = [
        np.log(nev[i + 1] / nev[i]) / np.log(ns[i + 1] / ns[i])
        for i in range(len(ns) - 1)
    ]
    return nev, loc


gauss = lambda n, d, s: np.random.default_rng(s).standard_normal(
    (n, d)
) / np.sqrt(d)
unif = lambda n, d, s: np.random.default_rng(s).random((n, d))

if __name__ == "__main__":
    for d, nu, lg, lu, ns in [
        (1, 1.5, 1.0, 0.2, [512, 1024, 2048, 4096, 6144]),
        (1, 2.5, 1.0, 0.2, [512, 1024, 2048, 4096, 6144]),
        (2, 2.5, 0.4, 0.15, [512, 1024, 2048, 4096]),
    ]:
        th = d / (2 * nu + d)
        print(f"Matern-{nu} d={d}  theta={th:.3f}")
        for name, gen, l in [("gauss", gauss, lg), ("unif", unif, lu)]:
            nev, loc = local_exponents(gen, d, nu, l, ns)
            print(
                f"  {name:6s} neff="
                + " ".join(f"{v:6.1f}" for v in nev)
                + "  local_theta="
                + " ".join(f"{e:.3f}" for e in loc)
            )
