from __future__ import division

import numpy as np
from sklearn.linear_model import Ridge


class linUCB:
    '''
    Implementation of the upper confidence bound algorithm for
    stochastic linear bandits
    '''
    def __init__(self, p, K, gamma_0, Sigma, lmbd, delta, rng, m2=1.3):
        '''
        Docstring for __init__
        :param p: parameter dimension
        :param K: number of actions
        :param gamma_0: true linear parameter
        :param Sigma: context-action covariance
        :param lmbd: regularization parameter
        :param delta: confidence parameter
        :param rng: random gen for reproducibility
        :param m2: parameter upper bound
        '''

        self.p = p
        self.K = K
        self.gamma_0 = gamma_0
        self.Sigma_0 = Sigma
        self.lmbd = lmbd
        self.rng = rng

        self.m2 = m2
        self.delta = delta
        self.t = 0

        # context initialization at t=0 and reward
        Xa_t = self.gen_ctx_act_features(self.p, self.K, self.rng)
        self.X = self.rng.choice(Xa_t,
                                 size=(1, ),
                                 axis=1,
                                 shuffle=False)
        self.true_R = self.reward(self.X)
        self.R = self.true_R + self.rng.normal()
        self.opt_R = self.opt_reward(Xa_t)
        self.gamma_hat = np.zeros((p, 1))
        self.Sigma = lmbd*np.eye(p) + np.outer(self.X, self.X.T)
        self.cov = self.X * self.R
        self.ridge = Ridge(alpha=self.lmbd, fit_intercept=False)
        self.alpha = self.m2 * np.sqrt(lmbd) + \
            np.sqrt(2 * np.log(1 / delta) + np.log(lmbd * p / lmbd**p))

    def gen_ctx_act_features(self, p, K, rng):
        # generates context-action features with given
        # covariance Sigma
        X = np.zeros((p, K))
        mu_x = np.zeros(p)
        for k in range(K):
            X_a = rng.multivariate_normal(mean=mu_x,
                                          cov=self.Sigma_0)
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

        # update true parameters if changed
        if not isinstance(gamma_0_t, bool):
            self.gamma_0 = gamma_0_t
        if not isinstance(Sigma, bool):
            self.Sigma_0 = Sigma
        if isinstance(Xa_t, bool):
            Xa_t = self.gen_ctx_act_features(self.p, self.K, self.rng)
        ucb = np.zeros((self.K, ))
        self.t += 1
        if self.X.shape[1] < 2:
            opt_a = self.rng.integers(0, self.K)
        else:
            self.gamma_hat = np.linalg.pinv(self.Sigma) @ self.cov
            self.alpha = self.m2 * np.sqrt(self.lmbd) \
                + np.sqrt(2 * np.log(1 / self.delta) + self.p *
                          np.log(1 + self.t / (self.lmbd * self.p)))

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
        self.Sigma += np.outer(X_opt, X_opt.T)
        self.cov += self.R[-1] * X_opt

        return
