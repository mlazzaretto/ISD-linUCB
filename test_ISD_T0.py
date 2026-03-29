from __future__ import division
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import ortho_group
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle

from bandits.ISDe_linUCB import ISDe_linUCB
from bandits.linUCB import linUCB
from bandits.ISD_linUCB import ISD_linUCB
from bandits.D_linUCB import D_linUCB
from bandits.sw_linUCB import sw_linUCB
from gen_hist_data import gen_data


def test_ISD_T0():
    '''
    Test ISD-linUCB for different values of T0
    comparison with linUCB, SW-linUCB, D-linUCB
    '''
    seed = 0
    rng = np.random.default_rng(seed)
    rng_isd = np.random.default_rng(seed)
    rng_sigma = np.random.default_rng(1)
    rng_sw = np.random.default_rng(seed)
    rng_d = np.random.default_rng(seed)

    p = 10
    K = 5
    T0_list = [1000, 3500, 8000]

    T = 500
    w = T       # single environment test
    delta = 1 / T
    lmbd = 0.1
    block_sizes = np.array([4, 1, 3, 2])
    blocks_c_gt = np.array([True, True, False, True])
    v_cfs = [5, 6, 7]
    c_cfs = [0, 1, 2, 3, 4, 8, 9]

    g = 0.999    # discounting factor

    n_iter = 20
    n_bandits = 6
    cum_reg = np.zeros((n_bandits, len(T0_list), n_iter))
    reg_rates = np.zeros((n_bandits, len(T0_list), n_iter, T))
    proj_errs = np.zeros((len(T0_list), n_iter, 2))

    U = ortho_group.rvs(dim=p, random_state=rng)

    for T0_idx, T0 in tqdm(enumerate(T0_list)):
        m = int(T0 / 8)
        it = 0

        while it < n_iter:
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

            A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
            mu_x = np.zeros(p)
            Sigma = U.T@A@A.T@U

            ISD_bandit = ISDe_linUCB(p, K, gamma_0,
                                     beta_inv, Sigma, lmbd,
                                     delta, w, X0, R0, rng, m2)

            blocks_est = np.array(ISD_bandit.blocks)
            blocks_c = np.array(ISD_bandit.c_blocks)
            sort_id_gt = np.argsort(block_sizes)
            sort_id_est = np.argsort(blocks_est)
            # only evaluate correctly estimated decompositon
            if (list(blocks_est[sort_id_est]) ==
                list(block_sizes[sort_id_gt]) and
                    list(blocks_c[sort_id_est]) ==
                    list(blocks_c_gt[sort_id_gt])):
                beta_inv_hat = ISD_bandit.beta_inv_hat
                print(blocks_est, blocks_c,
                      np.linalg.norm(beta_inv_hat - beta_inv))
                it += 1
            else:
                continue

            # evaulate projection error
            U_hat = ISD_bandit.U
            U_inv = U[list(c_cfs), :]
            U_res = U[list(v_cfs), :]
            c_cfs_est = ISD_bandit.cc
            v_cfs_est = ISD_bandit.vc
            U_inv_hat = U_hat[:, c_cfs_est]
            U_res_hat = U_hat[:, v_cfs_est]
            Pi_inv = U_inv.T@U_inv
            Pi_inv_hat = U_inv_hat@U_inv_hat.T
            P_inv_diff = Pi_inv - Pi_inv_hat
            Pi_res = U_res.T@U_res
            Pi_res_hat = U_res_hat@U_res_hat.T
            P_res_diff = Pi_res - Pi_res_hat
            proj_err_inv = np.linalg.norm(P_inv_diff, ord=2)
            proj_err_res = np.linalg.norm(P_res_diff, ord=2)
            proj_errs[T0_idx, it-1, 0] = proj_err_inv
            proj_errs[T0_idx, it-1, 1] = proj_err_res

            # other bandit algorithms for comparison
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
                ISD_bandit.update(Xa_t=Xa_t.T)
                for i_bdt, bandit in enumerate(bandits):
                    bandit.update(Xa_t=Xa_t.T)

            regret_ISD = ISD_bandit.opt_R - ISD_bandit.true_R.squeeze()

            for bandit in bandits:
                regrets.append(bandit.opt_R - bandit.true_R.squeeze())
                baselines.append(bandit.opt_R)
            # regret for oracle ISD, estimated beta_inv
            reg_res = bandits[-1].opt_R - bandits[-1].true_R_res.squeeze()
            regrets.append(reg_res)
            regrets.append(regret_ISD)

            for bn in range(n_bandits):
                reg = np.array(regrets[bn])
                reg_rate = reg[-T:]     # / np.sqrt(T)
                cum_reg[bn, T0_idx, it-1] = np.sum(reg_rate)
                reg_rates[bn, T0_idx, it-1, :] = reg_rate
    return cum_reg,  T0_list, reg_rates, proj_errs


def plot_reg_T0_comparison(cum_reg, T0_list, save_plot=False):
    # Plot cumulative regret for different T0s
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{mathrsfs}
        """,
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 1.5,
        'axes.unicode_minus': True,
    })

    offsets = np.linspace(-0.25, 0.25, 6)  # to separate violins for each group
    width = 0.2
    labels = ["linUCB",
              r"ISD-linUCB" + "\n" +
              r"(oracle $\mathcal{S}^{\text{inv}}$, " +
              r"$\mathcal{S}^{\text{res}}$)",
              r"ISD-linUCB" + "\n" +
              r"(oracle $\mathcal{S}^{\text{inv}}$, " +
              r"$\mathcal{S}^{\text{res}}$, $\beta^{\text{inv}}$)",
              "ISD-linUCB", "SW-linUCB", "D-linUCB",]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown"]

    fig, ax = plt.subplots(figsize=(8, 4))
    v1 = ax.violinplot(cum_reg[0, :, 1:].T,
                       showmeans=True,
                       showextrema=True,
                       positions=np.arange(len(T0_list))+offsets[0],
                       widths=width)
    v2 = ax.violinplot(cum_reg[3, :, 1:].T,
                       showmeans=True,
                       showextrema=True,
                       positions=np.arange(len(T0_list))+offsets[2],
                       widths=width)
    v3 = ax.violinplot(cum_reg[4, :, 1:].T,
                       showmeans=True,
                       showextrema=True,
                       positions=np.arange(len(T0_list))+offsets[3],
                       widths=width)
    v4 = ax.violinplot(cum_reg[5, :, 1:].T,
                       showmeans=True,
                       showextrema=True,
                       positions=np.arange(len(T0_list))+offsets[1],
                       widths=width)
    v5 = ax.violinplot(cum_reg[1, :, 1:].T,
                       showmeans=True,
                       showextrema=True,
                       positions=np.arange(len(T0_list))+offsets[4],
                       widths=width)
    v6 = ax.violinplot(cum_reg[2, :, 1:].T,
                       showmeans=True,
                       showextrema=True,
                       positions=np.arange(len(T0_list))+offsets[5],
                       widths=width)
    violin_sets = [v1, v2, v3, v4, v5, v6]
    markers = ['X', 'o', 'D', '^', 'P', '*']

    for v, c, m in zip(violin_sets, colors, markers):
        means = v['cmeans'].get_segments()
        xs = [seg.mean(axis=0)[0] for seg in means]
        ys = [seg.mean(axis=0)[1] for seg in means]

        ax.scatter(
            xs,
            ys,
            marker=m,
            s=40,
            facecolors=c,
            edgecolors=c,
            linewidths=1,
            zorder=3
        )
    for i in range(6):
        ax.add_line(
            plt.Line2D(
                [], [],
                color=colors[i],
                marker=markers[i],
                linestyle='None',
                markersize=6,
                markerfacecolor=colors[i],
                markeredgewidth=1,
                label=labels[i]
            )
        )
    for v in [v1, v2, v3, v4, v5, v6]:
        v['cmeans'].set_linewidth(0)
    ax.set_xticks(np.arange(len(T0_list)))
    ax.set_xticklabels(T0_list)
    ax.set_xlabel(r"$T_0$")
    ax.set_ylabel(r"$\sum_{t=1}^T\text{reg}_t$")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), labelspacing=1)
    ax.grid(visible=True, axis='y', color='gray', alpha=0.25, linestyle='-.')
    plt.tight_layout()
    dt = datetime.now()
    if not os.path.isdir('img/'+dt.strftime("%d%m%Y")):
        os.makedirs('img/'+dt.strftime("%d%m%Y"))
    if save_plot:
        plt.savefig('img/'+dt.strftime("%d%m%Y") +
                    '/T0T_test_T500_it20_keep_goodonly' +
                    dt.strftime("%H%M%S")+'.pdf', format='pdf',
                    bbox_inches='tight')
    plt.show()

    return


def plot_proj_err(proj_errs, T0_list, save_plot=False):
    plt.rcParams.update({
        "text.usetex": True,                # Use LaTeX for all text
        "font.family": "sans-serif",  # Use serif fonts (LaTeX default)
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{mathrsfs}
        """,
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 1.4,
        'axes.unicode_minus': True,
    })
    proj_errs_norm = proj_errs.copy()
    for j, T0 in enumerate(T0_list):
        proj_errs_norm[j, :] *= np.sqrt(T0)

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.8))
    ax[0].violinplot(proj_errs.T,
                     showmeans=True,
                     showextrema=True,
                     positions=np.arange(len(T0_list)),
                     # widths=width
                     )
    ax[1].violinplot(proj_errs_norm.T,
                     showmeans=True,
                     showextrema=True,
                     positions=np.arange(len(T0_list)),
                     # widths=width
                     )
    for j, axx in enumerate(ax):
        axx.set_xticks(np.arange(len(T0_list)))
        axx.set_xticklabels(T0_list)
        axx.set_xlabel(r"$T_0$")
        if j == 0:
            axx.set_ylabel(r"$\|\Pi^{\mathcal{S}^\text{inv}}-$" +
                           r"$\hat{\Pi}^{\mathcal{S}^\text{inv}}\|$" +
                           r"$_{\text{op}}$")
        else:
            axx.set_ylabel(r"$\sqrt{T_0}\|\Pi^{\mathcal{S}^\text{inv}}$" +
                           r"$-\hat{\Pi}^{\mathcal{S}^\text{inv}}\|" +
                           r"_{\text{op}}$")
        axx.grid(visible=True, axis='y', color='gray', alpha=0.25,
                 linestyle='-.')
    plt.tight_layout()
    if save_plot:
        dt = datetime.now()
        plt.savefig('img/'+dt.strftime("%d%m%Y") +
                    '/projerrs_T500_it20' +
                    dt.strftime("%H%M%S")+'.pdf', format='pdf',
                    bbox_inches='tight')
    plt.show()

    return


def main(run=False, plot_reg=False, save_rp=False,
         plot_proj=False, save_pp=False, save_data=False,
         filename=False):

    if run:
        cum_reg, T0_list, reg_rates, proj_errs = test_ISD_T0()
    else:
        with open(filename, 'rb') as file:
            cum_reg, T0_list, reg_rates, proj_errs = pickle.load(file)

    if save_data:
        dt = datetime.now()
        if not os.path.isdir('img/'+dt.strftime("%d%m%Y")):
            os.makedirs('img/'+dt.strftime("%d%m%Y"))
        with open('img/'+dt.strftime("%d%m%Y") +
                  '/data_500_20_go' + dt.strftime("%H%M%S") +
                  '.pkl', 'wb') as file:
            pickle.dump([cum_reg, T0_list, reg_rates, proj_errs], file)

    if plot_reg:
        plot_reg_T0_comparison(cum_reg, T0_list, save_rp)

    if plot_proj:
        plot_proj_err(proj_errs[:, :, 0], T0_list, save_pp)


if __name__ == '__main__':
    main(run=True, plot_reg=True, save_rp=True,
         plot_proj=True, save_pp=True, save_data=True,
         filename=False)
