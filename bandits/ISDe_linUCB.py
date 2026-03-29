from __future__ import division

import numpy as np
from isd.isd import ISD


class ISDe_linUCB:
    def __init__(self, p, K, gamma_0, beta_inv, Sigma,
                 lmbd, delta, w, X0, R0, rng, m2=1.3,
                 n_rw=25, ws_inv=None, k_fold=10, std=True):

        self.p = p  # context-action feature dimension
        self.K = K  # number of actions
        self.gamma_0 = gamma_0  # true bandit parameter
        self.beta_inv = beta_inv    # true invariant parameter
        self.lmbd = lmbd    # regularization parameter
        self.Sigma = Sigma
        self.rng = rng  # random number generator
        self.w = w  # window size for adaptation part
        self.X0 = X0  # past covariates observations
        self.R0 = R0  # past response observation

        self.m2 = m2     # bandit param. norm bound
        self.delta = delta  # confidence probability
        self.t = 0

        # Subspace and invariant parameter estimation
        self.T0 = X0.shape[0]
        self.n_rw = n_rw
        if ws_inv is None:
            self.ws_inv = int(self.T0 / 8)
        else:
            self.ws_inv = ws_inv
        est = ISD(X0, R0, [self.ws_inv]*self.n_rw)
        self.beta_inv_hat, \
            self.beta_icpt, \
            self.U, \
            self.blocks, \
            self.c_blocks = \
            est.invariant_estimator(k_fold=k_fold, std=std)
        self.U = self.U.T

        # Determine invariant and residual indices
        self.cc = []
        self.vc = []
        self.cf = False
        self.vf = False
        if self.c_blocks.any():
            self.cf = True
            be = np.cumsum(self.blocks)
            for j, b in enumerate(self.c_blocks):
                if b:
                    for idx in np.arange(be[j] - self.blocks[j], be[j]):
                        self.cc.append(idx)
                else:
                    self.vf = True
                    for idx in np.arange(be[j] - self.blocks[j], be[j]):
                        self.vc.append(idx)
        else:
            self.vf = True
            self.vc = list(np.arange(self.p))
        self.pinv = len(self.cc)
        self.pres = len(self.vc)

        # initialize with first round
        Xa_t = self.gen_ctx_act_features(self.p, self.K, self.rng)
        opt_a = np.argmax(Xa_t.T@self.beta_inv_hat)
        self.X = Xa_t[:, opt_a].reshape(-1, 1)
        self.X_inv = self.X
        self.true_R = self.reward(self.X)
        self.true_R_inv = self.X.T @  self.beta_inv
        eps = self.rng.normal()
        self.R = self.true_R + eps
        self.R_inv = self.true_R_inv + eps
        self.opt_R = self.opt_reward(Xa_t)
        self.opt_R_inv = self.opt_reward(Xa_t, inv=True)

        # init parameters estimates
        self.gamma_hat = np.zeros((p, 1))
        self.delta_res_hat = np.zeros((p, 1))
        self.Sigma_0 = self.X0.T @ self.X0
        self.lammax = np.linalg.eigh(self.Sigma_0 / self.T0)[0][-1]
        self.Delta_Pi = np.sqrt(self.lammax * np.log(2 * self.p / self.delta)
                                / self.T0)

        if self.cf:
            self.Sigma_inv_0 = self.U[:, self.cc].T @ \
                self.X0.T @ self.X0 @ \
                self.U[:, self.cc]
            self.cov_inv_0 = self.U[:, self.cc].T @ self.X0.T @ self.R0
            self.lam0 = np.linalg.eigh(self.Sigma_inv_0)[0][0]
            self.alpha_inv = np.sqrt(2 * np.log(1 / delta) + self.pinv *
                                     np.log(1 + 1 /
                                     (self.pinv * self.lam0)))
        else:
            self.alpha_inv = 0

        if self.vf:
            self.Sigma_res = (self.pres/self.p)*lmbd*np.eye(self.pres) + \
                self.U[:, self.vc].T @ np.outer(self.X, self.X.T) @ \
                self.U[:, self.vc]
            self.cov_res = (self.U[:, self.vc].T @ self.X) * self.R

    def gen_ctx_act_features(self, p, K, rng):
        # generates context-action features with given
        # covariance Sigma
        X = np.zeros((p, K))
        mu_x = np.zeros(p)
        for k in range(K):
            X_a = rng.multivariate_normal(mean=mu_x,
                                          cov=self.Sigma)
            X[:, k] = X_a
        return X

    def reward(self, X_t):
        return X_t.T @ self.gamma_0

    def opt_reward(self, Xa_t, inv=False):
        if inv:
            gamma = self.beta_inv
        else:
            gamma = self.gamma_0
        Ra = gamma.T @ Xa_t
        opt_a = np.argmax(Ra)
        return np.dot(gamma.T, Xa_t[:, opt_a])

    def update(self, gamma_0_t=False, Xa_t=False):
        # time t iteration
        # update estimated parameter and confidence set
        # choose action and compute reward
        if not isinstance(gamma_0_t, bool):
            self.gamma_0 = gamma_0_t
        if isinstance(Xa_t, bool):
            Xa_t = self.gen_ctx_act_features(self.p, self.K, self.rng)

        self.t += 1
        ucb = np.zeros((self.K, ))
        ucb_inv = np.zeros((self.K, ))
        if self.X.shape[1] < 2 and self.vf:
            opt_a = self.rng.integers(0, self.K)
            opt_a_inv = opt_a
        else:
            if self.vf:
                self.delta_res_hat = self.U[:, self.vc] @ \
                    np.linalg.solve(self.Sigma_res, self.cov_res)
            self.gamma_hat = self.beta_inv_hat + self.delta_res_hat
            if self.vf and self.cf:
                self.alpha = self.m2 * (self.pres/self.p) * np.sqrt(self.lmbd)\
                    + np.sqrt(2 * np.log(1 / self.delta)
                              + self.pinv
                              * np.log(1 + 1 /
                                       (self.pinv * self.lam0))
                              + self.pres *
                              np.log(1 + np.min([self.w, self.t]) / (self.lmbd
                                     * self.pres)))
            elif self.vf and not self.cf:
                self.alpha = self.m2 * np.sqrt(self.lmbd) \
                    + np.sqrt(2 * np.log(1 / self.delta)
                              + self.pres *
                              np.log(1 + np.min([self.w, self.t]) / (self.lmbd
                                     * self.pres)))
            else:
                self.alpha = self.alpha_inv

            for k in range(self.K):
                Xa = Xa_t[:, k].reshape(-1, 1)
                if self.cf:
                    S_inv_X = self.U[:, self.cc] @ \
                        np.linalg.solve(self.Sigma_inv_0,
                                        self.U[:, self.cc].T @ Xa)
                else:
                    S_inv_X = np.zeros((self.p, 1))
                if self.vf:
                    S_res_X = self.U[:, self.vc] @ \
                        np.linalg.solve(self.Sigma_res,
                                        self.U[:, self.vc].T @ Xa)
                else:
                    S_res_X = np.zeros((self.p, 1))
                ucb[k] = self.gamma_hat.T @ Xa \
                    + self.alpha * \
                    np.sqrt(Xa.T @ (S_inv_X + S_res_X))
                if self.cf:
                    ucb_inv[k] = self.beta_inv_hat.T @ Xa \
                        + self.alpha_inv * \
                        np.sqrt(Xa.T @ (self.U[:, self.cc] @
                                        np.linalg.solve(self.Sigma_inv_0,
                                                        self.U[:, self.cc].T
                                                        @ Xa)
                                        ))

            opt_a = np.argmax(ucb)
            if self.cf:
                opt_a_inv = np.argmax(ucb_inv)

        X_opt = Xa_t[:, opt_a].reshape(-1, 1)
        eps = self.rng.normal()

        if self.cf:
            X_opt_inv = Xa_t[:, opt_a_inv].reshape(-1, 1)
            self.X_inv = np.concatenate((self.X_inv, X_opt_inv), axis=1)
            self.true_R_inv = np.concatenate((self.true_R_inv,
                                              X_opt_inv.T @ self.beta_inv))
            self.R_inv = np.concatenate((self.R_inv,
                                         X_opt_inv.T @ self.beta_inv + eps))
        else:
            self.true_R_inv = np.concatenate((self.true_R_inv,
                                              np.zeros((1, 1))))
            self.R_inv = np.concatenate((self.R_inv,
                                         np.array([eps], ndmin=2)))
        self.X = np.concatenate((self.X, X_opt), axis=1)

        self.true_R = np.concatenate((self.true_R, self.reward(X_opt)))

        self.R = np.concatenate((self.R, self.reward(X_opt) + eps))

        self.opt_R = np.concatenate((self.opt_R, self.opt_reward(Xa_t)))
        self.opt_R_inv = np.concatenate((self.opt_R_inv,
                                         self.opt_reward(Xa_t, inv=True)))

        if self.vf:
            if self.R.shape[0] < self.w:
                self.Sigma_res += self.U[:, self.vc].T @ \
                    np.outer(X_opt, X_opt.T) @ \
                    self.U[:, self.vc]
                self.cov_res += (self.R[-1] - self.X[:, -1].T @
                                 self.beta_inv_hat) \
                    * self.U[:, self.vc].T @ X_opt

            else:
                self.Sigma_res = self.U[:, self.vc].T @ \
                    (self.X[:, -self.w:] @ self.X[:, -self.w:].T) @ \
                    self.U[:, self.vc] + \
                    self.lmbd * (self.pres / self.p) * np.eye(self.pres)
                self.cov_res = self.U[:, self.vc].T @ self.X[:, -self.w:] @ \
                    (self.R[-self.w:].reshape(-1, 1) - self.X[:, -self.w:].T
                     @ self.beta_inv_hat)

        return
