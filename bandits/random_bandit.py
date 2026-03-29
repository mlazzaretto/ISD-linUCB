from __future__ import division
import numpy as np


class random_bandit:
    def __init__(self, p, K, gamma_0, Sigma, rng):
        self.p = p
        self.K = K
        self.gamma_0 = gamma_0
        self.Sigma = Sigma
        self.rng = rng

        self.t = 0

        Xa_t = self.gen_ctx_act_features(self.p, self.K, self.Sigma, self.rng)
        self.X = self.rng.choice(Xa_t,
                                 size=(1, ),
                                 axis=1,
                                 shuffle=False)
        self.true_R = self.reward(self.X)
        self.R = self.true_R + 0.8*self.rng.normal()

    def gen_ctx_act_features(self, p, K, Sigma, rng):
        X = np.zeros((p, K))
        mu_x = np.zeros(p)
        for k in range(K):
            X_a = rng.multivariate_normal(mean=mu_x,
                                          cov=Sigma)
            X[:, k] = X_a
        return X

    def reward(self, X_t):
        return X_t.T @ self.gamma_0

    def update(self, gamma_0_t=False, Xa_t=False, Sigma=False):
        if not isinstance(gamma_0_t, bool):
            self.gamma_0 = gamma_0_t
        if not isinstance(Sigma, bool):
            self.Sigma = Sigma
        if isinstance(Xa_t, bool):
            Xa_t = self.gen_ctx_act_features(self.p,
                                             self.K,
                                             self.Sigma,
                                             self.rng)
        self.t += 1
        Ra = self.gamma_0.T @ Xa_t + 0.8*self.rng.normal(size=(1, self.K))
        if self.t % 2 == 0:
            opt_a = self.rng.integers(0, self.K)
        else:
            opt_a = np.argmax(Ra)
        X_opt = Xa_t[:, opt_a].reshape(-1, 1)
        self.X = np.concatenate((self.X, X_opt), axis=1)
        self.true_R = np.concatenate((self.true_R, Ra[:, [opt_a]]))
        self.R = np.concatenate((self.R, Ra[:, [opt_a]]+0.8*self.rng.normal()))
        return
