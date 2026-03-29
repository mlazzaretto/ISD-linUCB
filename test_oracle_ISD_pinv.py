from __future__ import division
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import ortho_group
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle

from bandits.linUCB import linUCB
from bandits.ISD_linUCB import ISD_linUCB
from bandits.D_linUCB import D_linUCB
from bandits.sw_linUCB import sw_linUCB
from gen_hist_data import gen_data


def test_dimension_oracle_sub():
    '''
    oracle subs
    fixed pres
    increasing pinv
    '''
    seed = 0
    rng = np.random.default_rng(seed)
    rng_isd = np.random.default_rng(seed)
    rng_sigma = np.random.default_rng(1)
    rng_sw = np.random.default_rng(seed)
    rng_d = np.random.default_rng(seed)

    P = np.arange(3, 11)
    K = 5
    T = 100
    T0 = 2000
    w = T       # single environment test
    delta = 1 / T
    lmbd = 0.1

    g = 0.999

    n_iter = 20
    n_bandits = 5
    cum_reg = np.zeros((n_bandits, P.shape[0], n_iter))
    reg_rates = np.zeros((n_bandits, P.shape[0], n_iter, T))

    for i, p in tqdm(enumerate(P)):
        m = int(T0 / 5)
        for j in range(n_iter):
            U = ortho_group.rvs(dim=p, random_state=rng)
            block_sizes = [2, p-2]
            c_cfs = list(np.arange(2, p))
            v_cfs = [0, 1]
            X0, R0, beta_inv = \
                gen_data(T0,
                         p,
                         m,
                         K,
                         block_sizes,
                         v_cfs,
                         U,
                         rng)

            gamma_0_til = np.zeros((p, 1))
            gamma_0_til[v_cfs, :] = rng.random((len(v_cfs), 1)) + 0.5
            gamma_0 = U.T@gamma_0_til + beta_inv
            m2 = np.sqrt(np.linalg.norm(gamma_0))

            A = block_diag(*[rng_sigma.random((bs, bs))
                             for bs in block_sizes])
            mu_x = np.zeros(p)
            Sigma = U.T@A@A.T@U

            bandits = [linUCB(p, K, gamma_0, Sigma, lmbd, delta, rng, m2),
                       sw_linUCB(p, K, gamma_0, Sigma, w, lmbd, delta, rng_sw),
                       D_linUCB(p, K, gamma_0, Sigma, lmbd, delta, g, rng_d),
                       ISD_linUCB(p, K, gamma_0, Sigma, lmbd, delta, w, U,
                                  c_cfs, v_cfs, rng_isd, X0, R0, m2)
                       ]

            regrets = []
            baselines = []

            for t in range(T):
                Xa_t = rng.multivariate_normal(mean=mu_x,
                                               cov=Sigma,
                                               size=K)
                for id_bdt, bandit in enumerate(bandits):
                    if (id_bdt == 1 or id_bdt == 2) and t == 0:
                        bandit.update(gamma_0_t=gamma_0, Xa_t=Xa_t.T)
                    else:
                        bandit.update(Xa_t=Xa_t.T)

            for bandit in bandits:
                regrets.append(bandit.opt_R - bandit.true_R.squeeze())
                baselines.append(bandit.opt_R)
            reg_res = bandits[-1].opt_R - bandits[-1].true_R_res.squeeze()
            regrets.append(reg_res)

            for bn in range(n_bandits):
                reg = np.array(regrets[bn])
                reg_rate = reg[-T:]
                cum_reg[bn, i, j] = np.sum(reg_rate)
                reg_rates[bn, i, j] = reg_rate
    return cum_reg, P, reg_rates, n_iter


def plot_oracle_ISD_pinv(cum_reg, P, save_plot=False):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{mathrsfs}
        """,
        'axes.labelsize': 13,
        'font.size': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'lines.linewidth': 1.3,
        'axes.unicode_minus': True,
    })

    offset = 0.1
    fig, ax = plt.subplots(figsize=(8, 3))
    v1 = ax.violinplot(cum_reg[0, :, :].T, showmeans=True,
                       positions=P-2*offset)
    v2 = ax.violinplot(cum_reg[3, :, :].T, showmeans=True,
                       positions=P-offset)
    v3 = ax.violinplot(cum_reg[1, :, :].T, showmeans=True,
                       positions=P+2*offset)
    v4 = ax.violinplot(cum_reg[2, :, :].T, showmeans=True,
                       positions=P+offset)

    # --------------------------------
    # Make mean segments invisible
    # --------------------------------
    v1['cmeans'].set_linewidth(0)
    v2['cmeans'].set_linewidth(0)
    v3['cmeans'].set_linewidth(0)
    v4['cmeans'].set_linewidth(0)

    # --------------------------------
    # Add mean markers
    # --------------------------------
    for v, m, c in zip([v1, v2, v3, v4], ['X', 'o', 'D', 'P'],
                       ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        means = v['cmeans'].get_segments()
        xs = [seg.mean(axis=0)[0] for seg in means]
        ys = [seg.mean(axis=0)[1] for seg in means]

        ax.scatter(
            xs,
            ys,
            marker=m,
            s=50,
            facecolors=c,
            edgecolors=c,
            linewidths=0.1,
            zorder=3
        )

    ax.set_xlabel('p')
    ax.set_xticks(P)
    ax.set_ylabel(r'$\sum_{t=1}^{T}\text{reg}_t$')
    ax.grid(color='grey', axis='both', linestyle='-.', linewidth=0.25,
            alpha=0.25)

    legend_handles = [
        plt.Line2D(
            [], [],
            marker='X',
            linestyle='None',
            color='tab:blue',
            markerfacecolor='tab:blue',
            markeredgewidth=1,
            markersize=6
        ),
        plt.Line2D(
            [], [],
            marker='o',
            linestyle='None',
            color='tab:orange',
            markerfacecolor='tab:orange',
            markeredgewidth=1,
            markersize=6
        ),
        plt.Line2D(
            [], [],
            marker='D',
            linestyle='None',
            color='tab:green',
            markerfacecolor='tab:green',
            markeredgewidth=1,
            markersize=6
        ),
        plt.Line2D(
            [], [],
            marker='P',
            linestyle='None',
            color='tab:red',
            markerfacecolor='tab:red',
            markeredgewidth=1,
            markersize=6
        )
    ]

    ax.legend(
        legend_handles,
        [r'linUCB',
         r'ISD-linUCB' + '\n' +
         r'(oracle $(\mathcal{S}^{\text{inv}}, \mathcal{S}^{\text{res}})$, ' +
         r'$p^{\text{res}}=2$)',
         'SW-linUCB', 'D-linUCB'],
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
    )

    plt.tight_layout()
    if save_plot:
        dt = datetime.now()
        plt.savefig('img/'+dt.strftime("%d%m%Y") +
                    '/reg_rates_pinv_comparison' +
                    dt.strftime("%H%M%S")+'.pdf', format='pdf',
                    bbox_inches='tight')
    plt.show()


def main(run=False, plot_reg=False, save_rp=False, save_data=False,
         filename=False):
    if run:
        cum_reg, P, reg_rates, n_iter = test_dimension_oracle_sub()
    else:
        with open(filename, 'rb') as file:
            cum_reg, P, reg_rates, n_iter = pickle.load(file)

    if save_data:
        dt = datetime.now()
        if not os.path.isdir('img/'+dt.strftime("%d%m%Y")):
            os.makedirs('img/'+dt.strftime("%d%m%Y"))
        with open('img/'+dt.strftime("%d%m%Y") +
                  '/data_pinv' + dt.strftime("%H%M%S") +
                  '.pkl', 'wb') as file:
            pickle.dump([cum_reg, P, reg_rates, n_iter], file)

    if plot_reg:
        plot_oracle_ISD_pinv(cum_reg, P, save_rp)


if __name__ == '__main__':
    filename = 'data_pinv164435.pkl'
    main(run=True, plot_reg=True, save_rp=True, save_data=True)
