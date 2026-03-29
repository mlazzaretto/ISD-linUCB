from __future__ import division
import numpy as np


class ISD_linUCB:
    def __init__(self, p, K, gamma_0, Sigma, lmbd, delta,
                 w, U, c_cfs, v_cfs, rng, X0=None, R0=None, m2=1.3,
                 restart=False):

        self.p = p  # context-action feature dimension
        self.K = K  # number of actions
        self.gamma_0 = gamma_0  # true bandit parameter
        self.Sigma = Sigma  # context-action covariance
        self.lmbd = lmbd    # regularization parameter
        self.rng = rng  # random number generator
        self.w = w  # window size for adaptation part
        self.ew = 1  # effective window size to simulate restarts
        self.U = U.T  # subspaces matrix
        self.vc = v_cfs     # invariant space coefficients
        self.cc = c_cfs     # residual space coefficients
        self.pinv = len(c_cfs)  # inv. space dimension
        self.pres = len(v_cfs)  # res. space dimension
        self.restart = restart  # restarting flag for res part
        self.X0 = X0
        self.R0 = R0
        if X0 is not None and R0 is not None:
            self.T0 = X0.shape[0]
            self.Sigma_0 = X0.T @ X0
            self.cov_0 = X0.T @ R0
            # estimated invariant component
            g_hat_pooled = np.linalg.solve(self.Sigma_0, self.cov_0)
            self.beta_inv_hat = self.U[:, self.cc] @ self.U[:, self.cc].T \
                @ g_hat_pooled

        # true invariant component
        self.beta_inv = self.U[:, self.cc] @ self.U[:, self.cc].T @ gamma_0

        self.m2 = m2     # bandit param. norm bound
        self.delta = delta  # confidence probability
        self.t = 0

        # initialize with first round
        Xa_t = self.gen_ctx_act_features(self.p, self.K, self.rng)
        if X0 is not None:
            opt_a = np.argmax(Xa_t.T@self.beta_inv_hat)
        else:
            opt_a = self.rng.choice(np.arange(K))

        eps = self.rng.normal()
        # Full ISD policy
        self.X = Xa_t[:, opt_a].reshape(-1, 1)
        self.true_R = self.reward(self.X)
        self.R = self.true_R + eps
        self.opt_R = self.opt_reward(Xa_t)

        self.X_opt_temp = np.zeros((self.p, 1))
        self.R_temp = np.zeros((1, ))

        # Invariant only policy
        self.X_inv = self.X
        self.true_R_inv = self.X.T @  self.beta_inv
        self.R_inv = self.true_R_inv + eps
        self.opt_R_inv = self.opt_reward(Xa_t, inv=True)

        # ISD with oracle inv. component policy
        self.X_res = self.X
        self.true_R_res = self.reward(self.X_res)
        self.R_res = self.true_R_res + eps

        # init parameters estimates
        # inv component - invariant only policy
        self.beta_inv_inv_hat = np.zeros((p, 1))
        # res component - full ISD policy
        self.delta_res_hat = np.zeros((p, 1))
        # res component - ISD with oracle inv
        self.delta_res_hat_or = np.zeros((p, 1))
        # gamma - ISD with oracle inv
        self.gamma_hat_or = np.zeros((p, 1))
        # should this be self.beta_inv.copy()?

        if X0 is not None:
            self.gamma_hat = self.beta_inv_hat.copy()
            self.Sigma_inv = U[:, self.cc].T @ X0.T @ X0 @ U[:, self.cc]
            self.cov_inv = (U[:, self.cc].T @ self.X0.T) @ self.R0
            self.lam0 = np.linalg.eigh(self.Sigma_inv)[0][0]
        else:
            self.gamma_hat = np.zeros((p, 1))
            self.beta_inv_hat = np.zeros((p, 1))
            self.Sigma_inv = (self.pinv/self.p)*lmbd*np.eye(self.pinv) + \
                U[:, self.cc].T @ np.outer(self.X, self.X.T) @ U[:, self.cc]
            self.cov_inv = (U[:, self.cc].T @ self.X) * self.R

        self.Sigma_res = (self.pres/self.p)*lmbd*np.eye(self.pres) + \
            U[:, self.vc].T @ np.outer(self.X, self.X.T) @ U[:, self.vc]

        self.Sigma_inv_inv = self.Sigma_inv.copy()
        self.Sigma_res_or = (self.pres/self.p)*lmbd*np.eye(self.pres) + \
            U[:, self.vc].T @ np.outer(self.X_res, self.X_res.T) @ \
            U[:, self.vc]

        self.cov_res = (U[:, self.vc].T @ self.X) * self.R
        self.cov_inv_inv = self.cov_inv.copy()
        self.cov_res_or = (U[:, self.vc].T @ self.X_res) * self.R_res

        self.alpha = self.m2 * np.sqrt(lmbd) + \
            np.sqrt(2 * np.log(1 / delta) + np.log(lmbd * p / lmbd**p))

        if X0 is not None:
            self.alpha_inv = np.sqrt(2 * np.log(1 / delta) + self.pinv *
                                     np.log(1 + 1 / (self.pinv *
                                            self.lam0)))
        else:
            self.alpha_inv = self.m2 * (self.pinv/self.p) * np.sqrt(lmbd) + \
                np.sqrt(2 * np.log(1 / delta) +
                        np.log(lmbd * self.pinv / lmbd**self.pinv))

        self.alpha_res = self.m2 * (self.pres/self.p) * np.sqrt(lmbd) + \
            np.sqrt(2 * np.log(1 / delta) +
                    np.log(lmbd * self.pres / lmbd**self.pres))

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

    def update(self, gamma_0_t=False, Xa_t=False, Sigma=False):
        # time t iteration
        # update estimated parameter and confidence set
        # choose action and compute reward
        if not isinstance(gamma_0_t, bool):
            self.gamma_0 = gamma_0_t
        if not isinstance(Sigma, bool):
            self.Sigma = Sigma
        if isinstance(Xa_t, bool):
            Xa_t = self.gen_ctx_act_features(self.p, self.K, self.rng)

        self.t += 1
        if self.t % self.w == 0:
            self.ew = 1
            bi_updt = True
        else:
            self.ew += 1
            bi_updt = False
        if self.restart:
            w = self.ew
        else:
            w = self.w

        ucb = np.zeros((self.K, ))
        ucb_inv = np.zeros((self.K, ))
        ucb_res = np.zeros((self.K, ))
        if self.X.shape[1] < 2 and self.X0 is None:
            opt_a = self.rng.integers(0, self.K)
            opt_a_inv = opt_a.copy()
            opt_a_res = opt_a.copy()
        elif self.X.shape[1] < 2 and self.X0 is not None:
            opt_a = np.argmax(Xa_t.T@self.beta_inv_hat)
            opt_a_inv = opt_a.copy()
            opt_a_res = np.argmax(Xa_t.T@self.beta_inv)
        else:
            # full ISD policy update
            if self.X0 is not None:
                if bi_updt:
                    self.beta_inv_hat = self.U[:, self.cc] @ \
                        self.U[:, self.cc].T @ \
                        np.linalg.solve(self.Sigma_0, self.cov_0)
                self.alpha = self.m2 * (self.pres/self.p) \
                    * np.sqrt(self.lmbd) \
                    + np.sqrt(2 * np.log(1 / self.delta)
                              + self.pinv *
                              np.log(1 + 1 / (self.lam0 * self.pinv))
                              + self.pres *
                              np.log(1 + np.min([w, self.t]) / (self.lmbd
                                     * self.pres)))
            else:
                self.beta_inv_hat = self.U[:, self.cc] @ \
                    np.linalg.solve(self.Sigma_inv, self.cov_inv)
                self.alpha = self.m2 * np.sqrt(self.lmbd) \
                    + np.sqrt(2 * np.log(1 / self.delta)
                              + self.pinv *
                              np.log(1 + self.t / (self.lmbd * self.pinv))
                              + self.pres *
                              np.log(1 + np.min([w, self.t]) / (self.lmbd
                                     * self.pres)))
                self.alpha_inv = self.m2 * (self.pinv/self.p) \
                    * np.sqrt(self.lmbd) \
                    + np.sqrt(2 * np.log(1 / self.delta)
                              + self.pinv *
                              np.log((1 + self.t / self.lmbd*self.pinv))
                              )
            self.delta_res_hat = self.U[:, self.vc] @ \
                np.linalg.solve(self.Sigma_res, self.cov_res)
            self.gamma_hat = self.beta_inv_hat + self.delta_res_hat

            # invaraint policy update
            self.beta_inv_inv_hat = self.U[:, self.cc] @ \
                np.linalg.solve(self.Sigma_inv_inv, self.cov_inv_inv)

            # ISD with oracle invariant comp. update
            self.delta_res_hat_or = self.U[:, self.vc] @ \
                np.linalg.solve(self.Sigma_res_or, self.cov_res_or)
            self.gamma_hat_or = self.beta_inv + self.delta_res_hat_or

            self.alpha_res = self.m2 * (self.pres/self.p) \
                * np.sqrt(self.lmbd) \
                + np.sqrt(2 * np.log(1 / self.delta)
                          + self.pres *
                          np.log(1 + np.min([w, self.t]) / (self.lmbd
                                 * self.pres))
                          )

            for k in range(self.K):
                Xa = Xa_t[:, k].reshape(-1, 1)
                ucb[k] = self.gamma_hat.T @ Xa \
                    + self.alpha * \
                    np.sqrt(Xa.T @ (self.U[:, self.cc] @
                                    np.linalg.solve(self.Sigma_inv,
                                                    self.U[:, self.cc].T @ Xa)
                                    + self.U[:, self.vc] @
                                    np.linalg.solve(self.Sigma_res,
                                                    self.U[:, self.vc].T @ Xa)
                                    ))
                ucb_inv[k] = self.beta_inv_inv_hat.T @ Xa \
                    + self.alpha_inv * \
                    np.sqrt(Xa.T @ (self.U[:, self.cc] @
                                    np.linalg.solve(self.Sigma_inv_inv,
                                                    self.U[:, self.cc].T @ Xa)
                                    ))
                ucb_res[k] = self.gamma_hat_or.T @ Xa \
                    + self.alpha_res * \
                    np.sqrt(Xa.T @ (self.U[:, self.vc] @
                                    np.linalg.solve(self.Sigma_res_or,
                                                    self.U[:, self.vc].T @ Xa)
                                    ))
            opt_a = np.argmax(ucb)
            opt_a_inv = np.argmax(ucb_inv)
            opt_a_res = np.argmax(ucb_res)

        X_opt = Xa_t[:, opt_a].reshape(-1, 1)
        X_opt_inv = Xa_t[:, opt_a_inv].reshape(-1, 1)
        X_opt_res = Xa_t[:, opt_a_res].reshape(-1, 1)

        self.X = np.concatenate((self.X, X_opt), axis=1)
        self.X_inv = np.concatenate((self.X_inv, X_opt_inv), axis=1)
        self.X_res = np.concatenate((self.X_res, X_opt_res), axis=1)
        self.true_R = np.concatenate((self.true_R, self.reward(X_opt)))
        self.true_R_inv = np.concatenate((self.true_R_inv,
                                          X_opt_inv.T @ self.beta_inv))
        self.true_R_res = np.concatenate((self.true_R_res,
                                          self.reward(X_opt_res)))

        eps = self.rng.normal()
        self.R = np.concatenate((self.R, self.reward(X_opt) + eps))
        self.R_inv = np.concatenate((self.R_inv,
                                     X_opt_inv.T @ self.beta_inv + eps))
        self.R_res = np.concatenate((self.R_res, self.reward(X_opt_res) + eps))

        self.opt_R = np.concatenate((self.opt_R, self.opt_reward(Xa_t)))
        self.opt_R_inv = np.concatenate((self.opt_R_inv,
                                         self.opt_reward(Xa_t, inv=True)))
        if bi_updt:
            self.Sigma_inv += self.U[:, self.cc].T @ \
                self.X_opt_temp @ self.X_opt_temp.T @ self.U[:, self.cc]
            self.cov_inv += (self.U[:, self.cc].T @
                             self.X_opt_temp) @ self.R_temp.reshape(-1, 1)
            self.X_opt_temp = np.zeros((self.p, 1))
            self.R_temp = np.zeros((1, ))
        else:
            self.X_opt_temp = np.concatenate((self.X_opt_temp, X_opt), axis=1)
            self.R_temp = np.concatenate((self.R_temp, self.R[-1]))

        self.Sigma_inv_inv += self.U[:, self.cc].T @ \
            np.outer(X_opt_inv, X_opt_inv.T) @ self.U[:, self.cc]
        self.cov_inv_inv += self.R_inv[-1] * self.U[:, self.cc].T \
            @ X_opt_inv

        if self.X0 is not None:
            self.Sigma_0 += np.outer(X_opt, X_opt.T)
            self.cov_0 += self.R[-1] * X_opt

        if self.R.shape[0] < self.w:
            self.Sigma_res += self.U[:, self.vc].T @ \
                np.outer(X_opt, X_opt.T) @ \
                self.U[:, self.vc]
            self.cov_res += (self.R[-1] - self.X[:, -1].T @ self.beta_inv_hat)\
                * self.U[:, self.vc].T @ X_opt
            self.Sigma_res_or += self.U[:, self.vc].T @ \
                np.outer(X_opt_res, X_opt_res.T) @ \
                self.U[:, self.vc]
            self.cov_res_or += (self.R_res[-1] - self.X_res[:, -1].T @
                                self.beta_inv) \
                * self.U[:, self.vc].T @ X_opt_res
        else:
            self.Sigma_res = self.U[:, self.vc].T @ \
                (self.X[:, -w:] @ self.X[:, -w:].T) @ \
                self.U[:, self.vc] + \
                self.lmbd * (self.pres / self.p) * np.eye(self.pres)
            self.cov_res = self.U[:, self.vc].T @ self.X[:, -w:] @ \
                (self.R[-w:].reshape(-1, 1) - self.X[:, -w:].T
                 @ self.beta_inv_hat)
            self.Sigma_res_or = self.U[:, self.vc].T @ \
                (self.X_res[:, -w:] @ self.X_res[:, -w:].T) @ \
                self.U[:, self.vc] + \
                self.lmbd * (self.pres / self.p) * np.eye(self.pres)
            self.cov_res_or = self.U[:, self.vc].T @ \
                self.X_res[:, -w:] @ \
                (self.R_res[-w:].reshape(-1, 1) -
                 self.X_res[:, -w:].T
                 @ self.beta_inv)
        return
