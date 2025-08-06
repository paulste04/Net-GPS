# ================================================================
#  Network-GPS Monte-Carlo with Quartile-specific DRF plots
#  ---------------------------------------------------------------

import numpy as np
import networkx as nx
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from patsy import dmatrix
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# ------------------- Configuration ----------------------------------------
config = {
    "n": 500,
    "p_edge": 0.10,
    "m_attach": 10,
    "rho": 0.7,
    "beta": 1.0,
    "sigma_T": 2.0,
    "n_sim": 100,
    "n_grid": 50,
    "alpha_T": 2,
    "sigma_U": 1,
    "sigma_H": 3,
    "second_stage_terms": ["WY_hat", "T", "R", "T*R", "T**2","R**2"],
    "dgp_terms": ["T", "R", "T*R"],
    "dgp_coeffs": {"T": 3, "R": 2, "T*R": 15},
    "alpha": 2,
    "norm_method": "row",
    "graph": "scalefree",
    "seed": 20250424,
}
np.random.seed(config["seed"])

EVAL_SCOPE = {"np": np, "sin": np.sin, "exp": np.exp, "log": np.log, "cos": np.cos,
              "tanh": np.tanh, "sqrt": np.sqrt}


# ------------------- Helper functions -------------------------------------
def eval_terms(terms, coeffs, variables):
    res = np.zeros_like(next(iter(variables.values())))
    for term in terms:
        res += coeffs[term] * eval(term, EVAL_SCOPE, variables)
    return res


def build_design_matrix(terms, variables):
    X_list = []
    for term in terms:
        if term.startswith("bs("):
            X_term = dmatrix(term, variables, return_type="dataframe")
            X_list.append(X_term.values)
        else:
            X_list.append(eval(term, {"np": np, "sin": np.sin, "exp": np.exp,
                                      "log": np.log}, variables))
    return np.column_stack(X_list)


def normalize(W, method="row"):
    if method == "row":
        row_sum = W.sum(1, keepdims=True)
        W_norm = np.divide(W, row_sum, where=row_sum != 0)
    elif method == "eigen":
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        W_norm = W / eig_max
    else:
        raise ValueError("Unsupported normalization method.")
    return np.nan_to_num(W_norm)


# ------------------- Core simulation function -----------------------------
def simulate_network(cfg):
    # ---------- Build network ------------------------------------------------
    if cfg['graph'] == 'erdos_renyi':
        G = nx.erdos_renyi_graph(config["n"], config["p_edge"])
    elif cfg['graph'] == 'scalefree':
        G = nx.barabasi_albert_graph(config["n"], config["m_attach"])    
    
    W = nx.to_numpy_array(G)
    W_row   = normalize(W, method="row")
    W_eigen = normalize(W, method="eigen")

    I = np.eye(cfg["n"])
    inv_row   = np.linalg.inv(I - cfg["rho"] * W_row)
    inv_eigen = np.linalg.inv(I - cfg["rho"] * W_eigen)

    # ---------- Degree quartiles --------------------------------------------
    degrees = np.asarray([d for _, d in G.degree()])
    quantiles = np.quantile(degrees, [0, .25, .50, .75, 1.0])
    deg_bin = np.digitize(degrees, quantiles[1:-1])          # 0,1,2,3
    masks = [(deg_bin == q).astype(float).reshape(-1, 1) for q in range(4)]
    denom = [m.sum() for m in masks]

    # ---------- Exogenous vars ----------------------------------------------
    H = np.random.normal(0, cfg["sigma_H"], (cfg["n"], 1))
    T = cfg["alpha_T"] + cfg["beta"] * H + np.random.normal(0, cfg["sigma_T"],
                                                            (cfg["n"], 1))

    grid = np.linspace(math.floor(T.min()), math.ceil(T.max()), cfg["n_grid"])
    rf = RandomForestRegressor(n_estimators=200, random_state=cfg["seed"])
    rf.fit(H, T.ravel())
    T_hat = rf.predict(H).reshape(-1, 1)
    sigma_T_hat = np.std(T - T_hat)
    R_obs = norm.pdf(T, loc=T_hat, scale=sigma_T_hat)
    TX_obs = H * R_obs

    # ---------- Instruments --------------------------------------------------
    def spatial_lags(Wmat, X, k=4):
        out = []
        Z = X
        for _ in range(k):
            Z = Wmat @ Z
            out.append(Z)
        return out

    instr_row   = [W_row @ H, *spatial_lags(W_row, H), W_row @ R_obs,
                   *spatial_lags(W_row, R_obs), W_row @ TX_obs]
    instr_eigen = [W_eigen @ H, *spatial_lags(W_eigen, H), W_eigen @ R_obs,
                   *spatial_lags(W_eigen, R_obs), W_eigen @ TX_obs]

    Z_row   = sm.add_constant(np.hstack(instr_row))
    Z_eigen = sm.add_constant(np.hstack(instr_eigen))

    # ---------- Containers ---------------------------------------------------
    n_sim, n_grid = cfg["n_sim"], cfg["n_grid"]
    mu_plugin_row   = np.zeros((n_sim, n_grid))
    mu_plugin_eigen = np.zeros_like(mu_plugin_row)
    mu_struct_row   = np.zeros_like(mu_plugin_row)
    mu_struct_eigen = np.zeros_like(mu_plugin_row)
    mu_naive_row     = np.zeros((config["n_sim"], config["n_grid"]))
    mu_naive_eigen   = np.zeros_like(mu_naive_row)         


    # quartile: 4 × g × s
    mu_plugin_q_row   = np.zeros((4, n_grid, n_sim))
    mu_plugin_q_eigen = np.zeros_like(mu_plugin_q_row)
    mu_struct_q_row   = np.zeros_like(mu_plugin_q_row)
    mu_struct_q_eigen = np.zeros_like(mu_plugin_q_row)

    # ---------- Monte-Carlo loop --------------------------------------------
    for s in range(n_sim):
        u = np.random.normal(0, cfg["sigma_U"], (cfg["n"], 1))
        eta = eval_terms(cfg["dgp_terms"], cfg["dgp_coeffs"], {"T": T, "R": R_obs})

        Y_row   = inv_row   @ (cfg["alpha"] + eta + u)
        Y_eigen = inv_eigen @ (cfg["alpha"] + eta + u)

        # 1st-stage: predict WY
        WY_row   = (W_row   @ Y_row).ravel()
        WY_eigen = (W_eigen @ Y_eigen).ravel()
        iv_row   = sm.OLS(WY_row,   Z_row).fit()
        iv_eigen = sm.OLS(WY_eigen, Z_eigen).fit()
        WY_hat_row   = iv_row.predict(Z_row).reshape(-1, 1)
        WY_hat_eigen = iv_eigen.predict(Z_eigen).reshape(-1, 1)

        # 2nd-stage regressions
        X2_row   = build_design_matrix(cfg["second_stage_terms"],
                                       {"T": T, "R": R_obs, "WY_hat": WY_hat_row})
        X2_eigen = build_design_matrix(cfg["second_stage_terms"],
                                       {"T": T, "R": R_obs, "WY_hat": WY_hat_eigen})
        X2_row   = sm.add_constant(X2_row,   has_constant="add")
        X2_eigen = sm.add_constant(X2_eigen, has_constant="add")
        b_row   = sm.OLS(Y_row.ravel(),   X2_row).fit().params
        b_eigen = sm.OLS(Y_eigen.ravel(), X2_eigen).fit().params

        inv_est_row   = np.linalg.inv(I - b_row[1]   * W_row)
        inv_est_eigen = np.linalg.inv(I - b_eigen[1] * W_eigen)
        b2_row   = np.asarray([x for j, x in enumerate(b_row)   if j != 1])
        b2_eigen = np.asarray([x for j, x in enumerate(b_eigen) if j != 1])

        # naive
        X_naive = build_design_matrix(["T", "T**2","T**3"], {"T": T})
        X_naive = sm.add_constant(X_naive, has_constant="add")
        mod_naive_row = sm.OLS(Y_row.ravel(), X_naive).fit()
        mod_naive_eigen = sm.OLS(Y_eigen.ravel(), X_naive).fit()    

        # ---------- Evaluate on grid -----------------------------------------
        for g, t_val in enumerate(grid):
            t_grid = np.full((cfg["n"], 1), t_val)
            r_grid = norm.pdf(t_grid, loc=T_hat, scale=sigma_T_hat)

            # prediction matrix w/o WY_hat
            X_pred = build_design_matrix(
                [t for t in cfg["second_stage_terms"] if "WY_hat" not in t],
                {"T": t_grid, "R": r_grid})
            X_pred = sm.add_constant(X_pred, has_constant="add")

            mu_row   = (inv_est_row   @ (X_pred @ b2_row)).mean()
            mu_eigen = (inv_est_eigen @ (X_pred @ b2_eigen)).mean()
            mu_plugin_row[s, g]   = mu_row
            mu_plugin_eigen[s, g] = mu_eigen

            X_pred_naive = build_design_matrix(["T", "T**2","T**3"], {"T": t_val})
            X_pred_naive = sm.add_constant(X_pred_naive, has_constant="add")

            mu_n_row   = (X_pred_naive @ mod_naive_row.params).mean()
            mu_n_eigen = (X_pred_naive @ mod_naive_eigen.params).mean()

            mu_naive_row[s, g]   = mu_n_row
            mu_naive_eigen[s, g] = mu_n_eigen


            # structural truth with fresh noise
            eta_grid = eval_terms(cfg["dgp_terms"], cfg["dgp_coeffs"],
                                  {"T": t_grid, "R": r_grid})
            Y_true_row   = inv_row   @ (cfg["alpha"] + eta_grid
                                        + np.random.normal(0, 1, (cfg["n"], 1)))
            Y_true_eigen = inv_eigen @ (cfg["alpha"] + eta_grid
                                        + np.random.normal(0, 1, (cfg["n"], 1)))
            mu_struct_row[s, g]   = Y_true_row.mean()
            mu_struct_eigen[s, g] = Y_true_eigen.mean()

            # -------- Quartile means -----------------------------------------
            for q, m in enumerate(masks):
                w = m / denom[q]
                mu_plugin_q_row[q, g, s]   = (w.T @ (inv_est_row   @
                                                     (X_pred @ b2_row))).item()
                mu_plugin_q_eigen[q, g, s] = (w.T @ (inv_est_eigen @
                                                     (X_pred @ b2_eigen))).item()
                mu_struct_q_row[q, g, s]   = (w.T @ Y_true_row).item()
                mu_struct_q_eigen[q, g, s] = (w.T @ Y_true_eigen).item()

    return (T, grid, mu_plugin_row, mu_plugin_eigen, mu_struct_row, mu_struct_eigen,
            mu_plugin_q_row, mu_plugin_q_eigen, mu_struct_q_row, mu_struct_q_eigen, mu_naive_row, mu_naive_eigen)

all_summaries = pd.DataFrame()


for size in list(range(50,501,50)):
    config['n']  = size
    for draw in range(config['n_sim']):
        # ======================  RUN + PLOTS ======================================
        (T, grid, mu_plug_row, mu_plug_eig, mu_str_row, mu_str_eig,
        mu_plug_q_row, mu_plug_q_eig, mu_str_q_row, mu_str_q_eig, mu_naive_row, mu_naive_eigen) = simulate_network(config)


        mu_structural_row = np.array(mu_str_row) 
        mu_plugin_row = np.array(mu_plug_row)

        mu_structural_eigen = np.array(mu_str_eig) 
        mu_plugin_eigen = np.array(mu_plug_eig)

        from sklearn.neighbors import KernelDensity

        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(T)
        log_dens = kde.score_samples(grid.reshape(-1, 1))
        dens = np.exp(log_dens)
        weights = dens / dens.sum()

        err_row   = mu_plugin_row - mu_structural_row       # plugin – structural
        err_eigen = mu_plugin_eigen - mu_structural_eigen

        bias_g_row = err_row.mean(axis=0)                 # Bias(t_g)
        mse_g_row  = (err_row**2).mean(axis=0)            # MSE(t_g)

        weighted_bias_row = np.dot(weights, bias_g_row)
        weighted_rmse_row = np.sqrt(np.dot(weights, mse_g_row))

        # ---- EIGEN normalisation ----------------------------------------
        bias_g_eig = err_eigen.mean(axis=0)
        mse_g_eig  = (err_eigen**2).mean(axis=0)

        weighted_bias_eig = np.dot(weights, bias_g_eig)
        weighted_rmse_eig = np.sqrt(np.dot(weights, mse_g_eig))
        ############## CONFIDENCE INTERVALS

        def weighted_bias_rmse(err, weights):
            """err shape (G, S) – grid rows, sims cols"""
            bias_g = err.mean(axis=0)         
            mse_g  = (err**2).mean(axis=0)
            wbias  = np.dot(weights, bias_g)
            wrmse  = np.sqrt(np.dot(weights, mse_g))
            return wbias, wrmse

        # ---- point estimates -----------------------------------------------------
        wbias_row,  wrmse_row  = weighted_bias_rmse(err_row,   weights)
        wbias_eig,  wrmse_eig  = weighted_bias_rmse(err_eigen, weights)

        # ---- bootstrap CI --------------------------------------------------------
        B = 1000
        wbias_boot_row  = np.empty(B)
        wrmse_boot_row  = np.empty(B)

        G, S = err_row.shape
        for b in range(B):
            idx = np.random.randint(0, S, size=S)           # resample columns
            err_boot = err_row[:, idx]                      # same shape (G, S)
            wbias_boot_row[b], wrmse_boot_row[b] = weighted_bias_rmse(err_boot, weights)

        # 95-percent bootstrap CIs
        ci_bias_row  = np.percentile(wbias_boot_row,  [2.5, 97.5])
        ci_rmse_row  = np.percentile(wrmse_boot_row,  [2.5, 97.5])

        # --- do the same for eigen ---
        wbias_boot_eig  = np.empty(B)
        wrmse_boot_eig  = np.empty(B)

        for b in range(B):
            idx = np.random.randint(0, S, size=S)
            err_boot = err_eigen[:, idx]
            wbias_boot_eig[b], wrmse_boot_eig[b] = weighted_bias_rmse(err_boot, weights)

        ci_bias_eig = np.percentile(wbias_boot_eig, [2.5, 97.5])
        ci_rmse_eig = np.percentile(wrmse_boot_eig, [2.5, 97.5])

        # Now for naive:

        # error arrays     (shape (G, S) if you transpose later)
        err_naive_row   = mu_naive_row   - mu_structural_row
        err_naive_eigen = mu_naive_eigen - mu_structural_eigen

        # weighted bias / RMSE (row example)
        wbias_naive_row, wrmse_naive_row = weighted_bias_rmse(err_naive_row, weights)
        wbias_naive_eigen, wrmse_naive_eigen = weighted_bias_rmse(err_naive_eigen, weights)


        # ------------------------------------------------------------------
        # 1)  ROW-normalised – naïve estimator
        # ------------------------------------------------------------------
        wbias_boot_naive_row  = np.empty(B)
        wrmse_boot_naive_row  = np.empty(B)

        for b in range(B):
            idx = np.random.randint(0, S, size=S)          # resample simulations
            err_boot = err_naive_row[:, idx]               # err_naive_row shape (G, S)
            wbias_boot_naive_row[b], wrmse_boot_naive_row[b] = weighted_bias_rmse(
                err_boot, weights)

        ci_bias_naive_row = np.percentile(wbias_boot_naive_row, [2.5, 97.5])
        ci_rmse_naive_row = np.percentile(wrmse_boot_naive_row, [2.5, 97.5])

        # ------------------------------------------------------------------
        # 2)  EIGEN-normalised – naïve estimator
        # ------------------------------------------------------------------
        wbias_boot_naive_eigen = np.empty(B)
        wrmse_boot_naive_eigen = np.empty(B)

        for b in range(B):
            idx = np.random.randint(0, S, size=S)
            err_boot = err_naive_eigen[:, idx]             # err_naive_eigen shape (G, S)
            wbias_boot_naive_eigen[b], wrmse_boot_naive_eigen[b] = weighted_bias_rmse(
                err_boot, weights)

        ci_bias_naive_eigen = np.percentile(wbias_boot_naive_eigen, [2.5, 97.5])
        ci_rmse_naive_eigen = np.percentile(wrmse_boot_naive_eigen, [2.5, 97.5])


        ###
        ###


        vars_needed = [
            "wbias_row", "wrmse_row", "wbias_naive_row", "wrmse_naive_row",
            "wbias_eig", "wrmse_eig", "wbias_naive_eigen", "wrmse_naive_eigen",
            "ci_bias_row", "ci_rmse_row", "ci_bias_naive_row", "ci_rmse_naive_row",
            "ci_bias_eig", "ci_rmse_eig", "ci_bias_naive_eigen", "ci_rmse_naive_eigen"
        ]

        globals_ns = globals()
        for v in vars_needed:
            if v not in globals_ns:
                globals_ns[v] = np.array([np.nan, np.nan]) if "ci_" in v else np.nan

        # ----------------------------------------------------------------------
        # Build the summary DataFrame
        # ----------------------------------------------------------------------
        summary = pd.DataFrame({
            "Metric": [
                "Weighted Bias (row, Net‑GPS)",
                "Weighted RMSE (row, Net‑GPS)",
                "Weighted Bias (row, Naive)",
                "Weighted RMSE (row, Naive)",
                "Weighted Bias (eigen, Net‑GPS)",
                "Weighted RMSE (eigen, Net‑GPS)",
                "Weighted Bias (eigen, Naive)",
                "Weighted RMSE (eigen, Naive)"
            ],
            "Point": [
                wbias_row,
                wrmse_row,
                wbias_naive_row,
                wrmse_naive_row,
                wbias_eig,
                wrmse_eig,
                wbias_naive_eigen,
                wrmse_naive_eigen
            ],
            "CI_lower": [
                ci_bias_row[0],
                ci_rmse_row[0],
                ci_bias_naive_row[0],
                ci_rmse_naive_row[0],
                ci_bias_eig[0],
                ci_rmse_eig[0],
                ci_bias_naive_eigen[0],
                ci_rmse_naive_eigen[0]
            ],
            "CI_upper": [
                ci_bias_row[1],
                ci_rmse_row[1],
                ci_bias_naive_row[1],
                ci_rmse_naive_row[1],
                ci_bias_eig[1],
                ci_rmse_eig[1],
                ci_bias_naive_eigen[1],
                ci_rmse_naive_eigen[1]
            ]
        })

        # Append to main summary
        summary = pd.concat([summary], ignore_index=True)
        summary['net_draw'] = draw
        summary['size'] = size
        all_summaries = all_summaries._append(summary)

        print(size)
        print(draw)
        
        


all_summaries.to_csv('C:/Users/Coding/Documents/ETH/Masterarbeit/Final/Outputs/network_sizes_output_sf.csv')

summary_er = pd.read_csv('C:/Users/Coding/Documents/ETH/Masterarbeit/Final/Outputs/network_sizes_output_sf.csv')
summary_sf = pd.read_csv('C:/Users/Coding/Documents/ETH/Masterarbeit/Final/Outputs/network_sizes_output.csv')


# Starting from your two dataframes
rmse_er = summary_er[summary_er["Metric"].str.contains("RMSE")].copy()
rmse_er["Network"] = "Erdős–Rényi"

rmse_sf = summary_sf[summary_sf["Metric"].str.contains("RMSE")].copy()
rmse_sf["Network"] = "Scale-Free"

# Combine the two
rmse_all = pd.concat([rmse_er, rmse_sf], ignore_index=True)

# Extract estimator and normalization method
rmse_all["Estimator"] = rmse_all["Metric"].apply(lambda x: "Naive" if "Naive" in x else "Net-GPS")
rmse_all["Normalization"] = rmse_all["Metric"].apply(lambda x: "Row" if "row" in x else "Eigen")

import seaborn as sns
import matplotlib.pyplot as plt

# Filter for Row normalization only
main_df = rmse_all[rmse_all["Normalization"] == "Row"].copy()

# Create boxplot per size
g = sns.catplot(
    data=main_df,
    kind="box",
    x="size",
    y="Point",
    hue="Estimator",
    col="Network",
    showfliers=False,  # Optional: hide outliers
    height=5,
    aspect=1.3,
    palette="Set2"
)

# Labeling
g.set_axis_labels("Network Size", "RMSE")
g.set_titles("{col_name} Network")
g._legend.set_title("Estimator")

# Tweak fonts
for ax in g.axes.flat:
    ax.tick_params(labelsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)

plt.tight_layout()
plt.show()

###############

g = sns.catplot(
    data=rmse_all,
    kind="box",
    x="size",
    y="Point",
    hue="Estimator",
    row="Normalization",
    col="Network",
    showfliers=False,
    height=4,
    aspect=1.3,
    palette="Set2"
)

# Axis labels
g.set_axis_labels("Network Size", "RMSE")
g.set_titles("{col_name} Network | {row_name} Normalization")
g._legend.set_title("Estimator")

# Font adjustments
for ax in g.axes.flat:
    ax.tick_params(labelsize=11)
    ax.set_xlabel(ax.get_xlabel(), fontsize=13)
    ax.set_ylabel(ax.get_ylabel(), fontsize=13)

plt.tight_layout()
plt.show()

# Ensure consistent category order for size
rmse_all["size"] = rmse_all["size"].astype(str)  # Convert to string for categorical treatment
size_order = sorted(rmse_all["size"].unique(), key=int)  # Sort numerically

# Plot with fixed category order
g = sns.catplot(
    data=rmse_all,
    kind="box",
    x="size",
    y="Point",
    hue="Estimator",
    row="Normalization",
    col="Network",
    order=size_order,                # Ensures same x-ticks across plots
    showfliers=False,
    height=4,
    aspect=1.3,
    palette="Set2"
)

# Set labels and titles
g.set_axis_labels("Network Size", "RMSE")
g.set_titles("{col_name} Network | {row_name} Normalization")
g._legend.set_title("Estimator")

# Improve font sizes
for ax in g.axes.flat:
    ax.tick_params(labelsize=11)
    ax.set_xlabel(ax.get_xlabel(), fontsize=13)
    ax.set_ylabel(ax.get_ylabel(), fontsize=13)

plt.tight_layout()
plt.show()