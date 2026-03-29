from __future__ import division
import numpy as np


class sw_linUCB:
    '''
    Implementation of the discounting algorithm  for non-stationary linear
    bandits presented in the paper
    Cheung, Wang Chi, David Simchi-Levi, and Ruihao Zhu. "Learning to optimize
    under non-stationarity." The 22nd International Conference on Artificial
    Intelligence and Statistics. PMLR, 2019.
    '''
    def __init__(self, p, K, gamma_0, Sigma, w, lmbd, delta, rng):

        self.p = p      # parameter dimension
        self.K = K      # number of actions
        self.gamma_0 = gamma_0  # true linear parameter
        self.Sigma_x = Sigma    # context-action covariance
        self.w = w              # window dimension
        self.lmbd = lmbd        # regularization parameter
        self.rng = rng          # random gen for reproducibility

        self.m2 = 1.3           # parameter upper bound
        self.delta = delta      # confidence parameter
        self.t = 0              # time counter

        # context initialization at t=0 and reward
        Xa_t = self.gen_ctx_act_features(self.p, self.K, self.Sigma_x,
                                         self.rng)
        self.X = self.rng.choice(Xa_t,
                                 size=(1, ),
                                 axis=1,
                                 shuffle=False)
        self.true_R = self.reward(self.X)
        self.R = self.true_R + self.rng.normal()
        self.opt_R = self.opt_reward(Xa_t)
        self.gamma_hat = np.zeros((p, 1))
        self.Sigma = lmbd*np.eye(p) + np.outer(self.X, self.X.T)
        self.cov = self.R * self.X
        self.alpha = self.m2 * np.sqrt(lmbd) + \
            np.sqrt(self.p *
                    np.log((1 + self.w / self.lmbd) / self.delta))

    def gen_ctx_act_features(self, p, K, Sigma, rng):
        # generates context-action features with given
        # covariance Sigma
        X = np.zeros((p, K))
        mu_x = np.zeros(p)
        for k in range(K):
            X_a = rng.multivariate_normal(mean=mu_x,
                                          cov=Sigma)
            X[:, k] = X_a
        return X

    def reward(self, X_t):
        # generate (noiseless) reward
        return X_t.T @ self.gamma_0

    def opt_reward(self, Xa_t):
        # generate optimal (noiseless) reward
        Ra = self.gamma_0.T @ Xa_t
        opt_a = np.argmax(Ra)
        return np.dot(self.gamma_0.T, Xa_t[:, opt_a])

    def update(self, gamma_0_t=False, Xa_t=False, Sigma=False):
        # time t iteration
        # updated estimated parameter and confidence set
        # chooses action and computes reward
        if not isinstance(gamma_0_t, bool):
            self.gamma_0 = gamma_0_t
        if not isinstance(Sigma, bool):
            self.Sigma_x = Sigma
        if isinstance(Xa_t, bool):
            Xa_t = self.gen_ctx_act_features(self.p, self.K, self.Sigma_x,
                                             self.rng)
        ucb = np.zeros((self.K, ))
        self.t += 1
        if self.X.shape[1] < 2:
            opt_a = self.rng.integers(0, self.K)
        else:
            self.gamma_hat = np.linalg.pinv(self.Sigma) @ self.cov
            self.alpha = self.m2 * np.sqrt(self.lmbd) \
                + np.sqrt(2 * np.log(1 / self.delta) + self.p *
                          np.log(1 + min(self.w, self.t) / (self.lmbd*self.p)))

            for k in range(self.K):
                Xa = Xa_t[:, k].reshape(-1, 1)
                ucb[k] = self.gamma_hat.T @ Xa \
                    + self.alpha * \
                    np.sqrt(Xa.T @ np.linalg.solve(self.Sigma, Xa))
            opt_a = np.argmax(ucb)
        X_opt = Xa_t[:, opt_a].reshape(-1, 1)
        self.X = np.concatenate((self.X, X_opt), axis=1)
        self.true_R = np.concatenate((self.true_R, self.reward(X_opt)))
        self.R = np.concatenate((self.R, self.reward(X_opt)+self.rng.normal()))
        self.opt_R = np.concatenate((self.opt_R, self.opt_reward(Xa_t)))
        if self.R.shape[0] < self.w:
            self.Sigma += np.outer(X_opt, X_opt.T)
            self.cov += self.R[-1] * X_opt

        else:
            self.Sigma = self.X[:, -self.w:] @ self.X[:, -self.w:].T \
                + self.lmbd * np.eye(self.p)
            self.cov = self.X[:, -self.w:] @ self.R[-self.w:].reshape(-1, 1)

        return
