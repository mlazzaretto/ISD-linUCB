from __future__ import division
import numpy as np
from scipy.linalg import block_diag
from bandits.random_bandit import random_bandit


def gen_data(T0, p, m, K, block_sizes, v_coeffs, U, rng):
    # generate historical data
    # T0: sample size
    # p: covariates dimension
    # m: number of covariate shifts in T0
    # k: action space dimension
    # block_sizes: list of irreducible subspaces dimensions
    # c_coeffs: list of indices of constant coeffs
    # U: orthonormal matrix aligned with irreducible subs
    # rng: random generator

    rng_sigma = np.random.default_rng(42)
    rng_gamma = np.random.default_rng(0)

    # initialize linear parameter
    gamma_0_til = rng_gamma.random((p, 1)) + 0.5
    gamma_0_til /= np.linalg.norm(gamma_0_til)
    gamma_0 = U.T@gamma_0_til
    beta_inv = gamma_0_til.copy()
    beta_inv[v_coeffs, :] = 0
    beta_inv = U.T @ beta_inv

    A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
    Sigma = U.T @ A @ A.T @ U

    bandit = random_bandit(p, K, gamma_0, Sigma, rng)

    gamma_0_mat = np.zeros((T0 + 1, p))
    gamma_0_mat[0, :] = gamma_0_til.squeeze()
    for t in range(T0):
        for j in v_coeffs:
            gamma_0_til[j, :] = \
                gamma_0_mat[0, j]-1.5*(t/T0)*(np.sin((j+1)*0.25*t/T0+(j+1))**2)
            gamma_0 = U.T@gamma_0_til

        if t % m == 0:
            A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
            Sigma = U.T @ A @ A.T @ U

            bandit.update(gamma_0_t=gamma_0, Sigma=Sigma)
        else:
            bandit.update(gamma_0_t=gamma_0)
        gamma_0_mat[t + 1, :] = gamma_0_til.squeeze()

    return bandit.X.T, bandit.R, beta_inv
