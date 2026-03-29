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
from gen_hist_data import gen_data


def test_ISD_oracle_pres():
    '''
    test ISD-linUCB (oracle subspaces) for different values of p^res
    '''
    seed = 0
    rng = np.random.default_rng(seed)
    rng_isd = np.random.default_rng(seed)
    rng_sigma = np.random.default_rng(1)

    p = 10
    K = 5
    T = 100
    T0_list = [2000,]
    w = T       # single environment test
    delta = 1 / T
    lmbd = 0.1
    block_sizes = np.array([4, 1, 3, 2])
    v_cfs_list = [[8, 9], [0, 1, 2, 3], [4, 5, 6, 7, 8, 9],
                  [0, 1, 2, 3, 4, 5, 6, 7]]
    c_cfs_list = [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9], [0, 1, 2, 3],
                  [8, 9]]

    n_iter = 20
    n_bandits = 3
    n_ps = len(v_cfs_list)
    cum_reg = np.zeros((n_bandits, len(T0_list), n_ps, n_iter))
    reg_rates = np.zeros((n_bandits, len(T0_list), n_ps, n_iter, T))

    U = ortho_group.rvs(dim=p, random_state=rng)

    for T0_idx, T0 in tqdm(enumerate(T0_list)):
        m = int(T0 / 5)
        for i_cfs in range(len(v_cfs_list)):
            print(i_cfs)
            v_cfs = v_cfs_list[i_cfs]
            c_cfs = c_cfs_list[i_cfs]

            it = 0
            while it < n_iter:
                it += 1
                X0, R0, beta_inv = gen_data(T0,
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
                           ISD_linUCB(p, K, gamma_0, Sigma, lmbd, delta, w, U,
                                      c_cfs, v_cfs, rng_isd, X0, R0, m2)
                           ]
                regrets = []
                baselines = []

                for t in range(T):
                    Xa_t = rng.multivariate_normal(mean=mu_x,
                                                   cov=Sigma,
                                                   size=K)
                    for bandit in bandits:
                        bandit.update(Xa_t=Xa_t.T)

                for bandit in bandits:
                    regrets.append(bandit.opt_R - bandit.true_R.squeeze())
                    baselines.append(bandit.opt_R)
                reg_res = bandits[-1].opt_R - bandits[-1].true_R_res.squeeze()
                regrets.append(reg_res)

                for bn in range(n_bandits):
                    reg = np.array(regrets[bn])
                    reg_rate = reg[-T:]     # / np.sqrt(T)
                    cum_reg[bn, T0_idx, i_cfs, it-1] = np.sum(reg_rate)
                    reg_rates[bn, T0_idx, i_cfs, it-1, :] = reg_rate
    return cum_reg, T0_list, v_cfs_list, reg_rates


def plot_reg_pres_oracle_comparison_full_hor(reg_rates, save_plt,
                                             pres_list=None):
    # plotting for 'test_ISD_oracle_T0'
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

    n_methods, n_T0, n_dim, n_reps, T = reg_rates.shape
    reg_rates_cs = np.cumsum(reg_rates, axis=-1)
    t = np.arange(1, T + 1)

    fig, (ax, ax_v) = plt.subplots(
        ncols=2,
        figsize=(6.5, 2.5), sharey=True
    )

    # Store handles + violin data
    h_linucb = None
    h_isd = []

    violin_data = []
    violin_colors = []

    for j in range(n_T0):

        if j == 0:
            # ---- linUCB
            mean_curve = reg_rates_cs[0, j, -1].mean(axis=0)
            std_curve = reg_rates_cs[0, j, -1].std(axis=0)
            h_linucb, = ax.plot(t, mean_curve)
            ax.fill_between(
                t,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
                color=h_linucb.get_color()
            )

        # ---- ISD (one per p_res)
        for p_idx in range(n_dim):
            mean_curve = reg_rates_cs[1, j, p_idx].mean(axis=0)
            std_curve = reg_rates_cs[1, j, p_idx].std(axis=0)
            h, = ax.plot(t, mean_curve)
            ax.fill_between(
                t,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
                color=h.get_color()
            )
            h_isd.append(h)

            violin_data.append(reg_rates_cs[1, j, p_idx][:, -1])
            violin_colors.append(h.get_color())
        violin_data.append(reg_rates_cs[0, j, -1][:, -1])
        violin_colors.append(h_linucb.get_color())

    # -------------------------------
    # Legend
    # -------------------------------
    text_only = plt.Line2D(
        [], [],
        color=h_isd[0].get_color(),
        linewidth=h_isd[0].get_linewidth(),
        alpha=0
    )

    spacer = plt.Line2D([], [], linestyle='none')

    ax.set_xlabel("t")
    ax.set_ylabel(
        r"$\sum_{t=1}^T\text{reg}_t$"
    )
    ax.grid(True, alpha=0.25, linestyle='-.')

    # ===============================
    # Violin subplot
    # ===============================
    vp = ax_v.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    for body, color in zip(vp['bodies'], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)

    for part in ['cmins', 'cmaxes', 'cbars']:
        vp[part].set_color(violin_colors)
        vp[part].set_linewidth(1.4)
    for part in ['cmeans']:
        vp[part].set_color(violin_colors)
        vp[part].set_linewidth(0.0)

        # One marker per unique color
    unique_colors = list(dict.fromkeys(violin_colors))
    markers = ['o', 'D', '^', 'P', 'X', 'v', 's']
    color_to_marker = {
        c: markers[i % len(markers)]
        for i, c in enumerate(unique_colors)
    }

    # Compute means and overlay markers
    violin_means = [np.mean(v) for v in violin_data]
    positions = np.arange(1, len(violin_data) + 1)

    for x, y, c in zip(positions, violin_means, violin_colors):
        ax_v.scatter(
            x,
            y,
            marker=color_to_marker[c],
            s=40,
            facecolors=c,
            edgecolors=c,
            linewidths=1,
            zorder=3
        )
    legend_marker_handles = []
    # linUCB marker
    legend_marker_handles.append(
        plt.Line2D(
            [], [],
            color=h_linucb.get_color(),
            marker=color_to_marker[h_linucb.get_color()],
            linestyle='-',
            markersize=6,
            markerfacecolor=c,
            markeredgewidth=1
        )
    )

    # ISD markers
    for h in h_isd:
        c = h.get_color()
        legend_marker_handles.append(
            plt.Line2D(
                [], [],
                color=c,
                marker=color_to_marker[c],
                linestyle='-',
                markersize=6,
                markerfacecolor=c,
                markeredgewidth=1
            )
        )

    legend_handles = [
        legend_marker_handles[0], spacer,
        text_only, spacer,
        *legend_marker_handles[1:]
    ]

    legend_labels = [
        "linUCB", "",
        r"ISD-linUCB" + r"(oracle $\mathcal{S}^{\mathrm{inv}}$, "
        r"$\mathcal{S}^{\mathrm{res}}$):", "",
        *[r"$p^{\text{res}}=" + f"{p}$" for p in pres_list]
    ]

    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.1),
        frameon=True,
        ncol=4
    )

    ax_v.set_xticks([])
    ax_v.grid(True, axis='y', alpha=0.4, linestyle='-.')

    fig.tight_layout()
    if save_plt:
        dt = datetime.now()
        if not os.path.isdir('img/'+dt.strftime("%d%m%Y")):
            os.makedirs('img/'+dt.strftime("%d%m%Y"))
        plt.savefig('img/'+dt.strftime("%d%m%Y") +
                    '/pres_regret_comparison_lines' +
                    dt.strftime("%H%M%S")+'.pdf',
                    format='pdf', bbox_inches='tight')
    plt.show()


def main(run=False, plot_reg=False, save_rp=False, save_data=False,
         filename=False):

    if run:
        cum_reg, T0_list, v_cfs_list, reg_rates = test_ISD_oracle_pres()
    else:
        with open(filename, 'rb') as file:
            cum_reg, T0_list, v_cfs_list, reg_rates = pickle.load(file)

    if save_data:
        dt = datetime.now()
        if not os.path.isdir('img/'+dt.strftime("%d%m%Y")):
            os.makedirs('img/'+dt.strftime("%d%m%Y"))
        with open('img/'+dt.strftime("%d%m%Y") +
                  '/data_pres' + dt.strftime("%H%M%S") +
                  '.pkl', 'wb') as file:
            pickle.dump([cum_reg, T0_list, v_cfs_list, reg_rates], file)

    if plot_reg:
        plot_reg_pres_oracle_comparison_full_hor(
            reg_rates, save_rp,
            pres_list=[len(clist) for
                       clist in v_cfs_list])


if __name__ == '__main__':
    main(run=True, plot_reg=True, save_rp=True, save_data=True)
