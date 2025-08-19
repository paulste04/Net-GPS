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
import random

# ------------------- Configuration ----------------------------------------
config = {
    "n": 500,
    "p_edge": 0.10,
    "m_attach": 10,
    "rho": 0.7,
    "beta": 1.0,
    "sigma_T": 2.0,
    "n_sim": 500,
    "n_grid": 50,
    "alpha_T": 2,
    "sigma_U": 1,
    "sigma_H": 3,
    "second_stage_terms": ["WY_hat", "T", "R", "T**2", "R**2", "T*R"],
    "dgp_terms": ["T", "R", "T*R"],
    "dgp_coeffs": {"T": 3, "R": 2, "T*R": 15},
    "alpha": 2,
    "norm_method": "eigen",
    "graph": "scalefree",
    "seed": 202507,
}
np.random.seed(config["seed"])
random.seed(config['seed'])


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

    # W_eigen = pd.read_csv('C:/Users/psfin/Documents/W_eigen_er.csv').to_numpy() # for ensuring full consistency with local estimand manual import of normalized matrix 
    # W_row = pd.read_csv('C:/Users/psfin/Documents/W_row_er.csv').to_numpy()
    # W_eigen = pd.read_csv('C:/Users/psfin/Documents/W_eigen_sf.csv').to_numpy()
    # W_row = pd.read_csv('C:/Users/psfin/Documents/W_row_sf.csv').to_numpy()

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
            mu_plugin_q_row, mu_plugin_q_eigen, mu_struct_q_row, mu_struct_q_eigen, mu_naive_row, mu_naive_eigen, W_eigen_2, W_eigen)


# ======================  RUN + PLOTS ======================================
(T, grid, mu_plug_row, mu_plug_eig, mu_str_row, mu_str_eig,
 mu_plug_q_row, mu_plug_q_eig, mu_str_q_row, mu_str_q_eig, mu_naive_row, mu_naive_eigen,W_eigen_2, W_eigen) = simulate_network(config)

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
B = 500
wbias_boot_row  = np.empty(B)
wrmse_boot_row  = np.empty(B)

G, S = err_row.shape  # shape: 500 (node size) x 50 (simulation runs)
for b in range(B):
    idx = np.random.randint(0, G, size=G)           # resample columns
    err_boot = err_row[idx, :]     
    wbias_boot_row[b], wrmse_boot_row[b] = weighted_bias_rmse(err_boot, weights)

# 95-percent bootstrap CIs
ci_bias_row  = np.percentile(wbias_boot_row,  [2.5, 97.5])
ci_rmse_row  = np.percentile(wrmse_boot_row,  [2.5, 97.5])

# --- do the same for eigen ---
wbias_boot_eig  = np.empty(B)
wrmse_boot_eig  = np.empty(B)

for b in range(B):
    idx = np.random.randint(0, G, size=G)
    err_boot = err_eigen[idx, :]
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
    idx = np.random.randint(0, G, size=G)          # resample simulations
    err_boot = err_naive_row[idx, :]               # err_naive_row shape (G, S)
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
    idx = np.random.randint(0, G, size=G)
    err_boot = err_naive_eigen[idx, :]             # err_naive_eigen shape (G, S)
    wbias_boot_naive_eigen[b], wrmse_boot_naive_eigen[b] = weighted_bias_rmse(
        err_boot, weights)

ci_bias_naive_eigen = np.percentile(wbias_boot_naive_eigen, [2.5, 97.5])
ci_rmse_naive_eigen = np.percentile(wrmse_boot_naive_eigen, [2.5, 97.5])


### Gradient bias ###

# Compute gradients along treatment axis
grad_true_row = np.gradient(mu_structural_row, axis=1)  # shape (n_sim, n_grid)
grad_est_row  = np.gradient(mu_plugin_row,  axis=1)
grad_naive_row = np.gradient(mu_naive_row, axis=1)

grad_true_eigen = np.gradient(mu_structural_eigen, axis=1)  # shape (n_sim, n_grid)
grad_est_eigen  = np.gradient(mu_plugin_eigen,  axis=1)
grad_naive_eigen = np.gradient(mu_naive_eigen, axis=1)


# Compute gradient bias matrix
grad_bias_row = grad_est_row - grad_true_row  # shape (n_sim, n_grid)
grad_bias_naive_row = grad_est_row - grad_naive_row

grad_bias_eigen = grad_est_eigen - grad_true_eigen  # shape (n_sim, n_grid)
grad_bias_naive_eigen = grad_est_eigen - grad_naive_eigen


# Average across sims and grid

mean_grad_bias_row, rmse_grad_bias_row = weighted_bias_rmse(grad_bias_row, weights)
mean_grad_bias_naive_row, rmse_grad_bias_naive_row = weighted_bias_rmse(grad_bias_naive_row, weights)

mean_grad_bias_eigen, rmse_grad_bias_eigen = weighted_bias_rmse(grad_bias_eigen, weights)
mean_grad_bias_naive_eigen, rmse_grad_bias_naive_eigen = weighted_bias_rmse(grad_bias_naive_eigen, weights)


# Bootstrap CI
def get_grad_ci(grad_bias):
    B = 500
    n_sim = grad_bias.shape[0]
    boot_bias, boot_rmse = [], []

    for _ in range(B):
        idx = np.random.choice(n_sim, size=n_sim, replace=True)
        sample = grad_bias[idx, :]
        boot_bias_new, boot_rmse_new = weighted_bias_rmse(
        sample, weights)

        boot_bias.append(boot_bias_new)
        boot_rmse.append(boot_rmse_new)

    ci_bias = np.percentile(boot_bias, [2.5, 97.5])
    ci_rmse = np.percentile(boot_rmse, [2.5, 97.5])
    return(ci_bias, ci_rmse)

grad_ci_bias_row, grad_ci_rmse_row = get_grad_ci(grad_bias_row)
grad_ci_bias_naive_row, grad_ci_rmse_naive_row = get_grad_ci(grad_bias_naive_row)

grad_ci_bias_eigen, grad_ci_rmse_eigen = get_grad_ci(grad_bias_eigen)
grad_ci_bias_naive_eigen, grad_ci_rmse_naive_eigen = get_grad_ci(grad_bias_naive_eigen)


# Output



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

gradient_df = pd.DataFrame({
    "Metric": [
        "Gradient Bias (row, Net‑GPS)",
        "Gradient RMSE (row, Net‑GPS)",
        "Gradient Bias (row, Naive)",
        "Gradient RMSE (row, Naive)",
        "Gradient Bias (eigen, Net‑GPS)",
        "Gradient RMSE (eigen, Net‑GPS)",
        "Gradient Bias (eigen, Naive)",
        "Gradient RMSE (eigen, Naive)"
    ],
    "Point": [
        mean_grad_bias_row,
        rmse_grad_bias_row,
        mean_grad_bias_naive_row,
        rmse_grad_bias_naive_row,
        mean_grad_bias_eigen,
        rmse_grad_bias_eigen,
        mean_grad_bias_naive_eigen,
        rmse_grad_bias_naive_eigen
    ],
    "CI_lower": [
        grad_ci_bias_row[0],
        grad_ci_rmse_row[0],
        grad_ci_bias_naive_row[0],
        grad_ci_rmse_naive_row[0],
        grad_ci_bias_eigen[0],
        grad_ci_rmse_eigen[0],
        grad_ci_bias_naive_eigen[0],
        grad_ci_rmse_naive_eigen[0]
    ],
    "CI_upper": [
        grad_ci_bias_row[1],
        grad_ci_rmse_row[1],
        grad_ci_bias_naive_row[1],
        grad_ci_rmse_naive_row[1],
        grad_ci_bias_eigen[1],
        grad_ci_rmse_eigen[1],
        grad_ci_bias_naive_eigen[1],
        grad_ci_rmse_naive_eigen[1]
    ]
})

# Append to main summary
summary = pd.concat([summary, gradient_df], ignore_index=True)
summary.to_csv('C:/Users/psfin/Documents/Thesis_Results/results_global_'+config["graph"]+'_'+config['norm_method']+'.csv',index=False)
############
#############
###############
################
##############
##################


# ---------- Collapse across simulations -----------------------------------
def mean_lo_hi(arr):
    return arr.mean(0), np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0)

m_row, lo_row, hi_row = mean_lo_hi(mu_plug_row)
s_row, slo_row, shi_row = mean_lo_hi(mu_str_row)

# ---------- Global DRF plot -----------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(grid, s_row, lw=2, label="Structural (row-norm.)")
plt.plot(grid, m_row, "--", lw=2, label="Net-GPS (row-norm.)")
plt.fill_between(grid, lo_row, hi_row, alpha=.15, color = 'orange')
plt.title("Global Dose-Response – Row-normalised W")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\mu(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
#              DEGREE-QUARTILE DOSE-RESPONSE PLOT (row-normalised)
# --------------------------------------------------------------------------
quartile_labels = ["Q1: lowest deg.", "Q2", "Q3", "Q4: hubs"]
colors = ["steelblue", "forestgreen", "goldenrod", "firebrick"]

# ---- collapse over simulation draws (axis=2) for quartile matrices -------
mu_str_q_mean   = mu_str_q_row.mean(axis=2)            # 4 × n_grid
mu_plug_q_mean  = mu_plug_q_row.mean(axis=2)
lo_q            = np.percentile(mu_plug_q_row,  2.5, axis=2)
hi_q            = np.percentile(mu_plug_q_row, 97.5, axis=2)

plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(grid, mu_str_q_mean[q], color=colors[q], lw=2,
             label=f"Structural {quartile_labels[q]}")
    plt.plot(grid, mu_plug_q_mean[q], "--", color=colors[q], lw=2)
    plt.fill_between(grid, lo_q[q], hi_q[q], color=colors[q], alpha=.15)
plt.title("Row-normalised DRF by Degree Quartile")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\mu_q(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
#              DEGREE-QUARTILE DOSE-RESPONSE PLOT (eigen-normalised)
# --------------------------------------------------------------------------
# Collapse over simulation draws (axis = 2)
mu_str_q_eig_mean  = mu_str_q_eig.mean(axis=2)        # 4 × n_grid
mu_plug_q_eig_mean = mu_plug_q_eig.mean(axis=2)
lo_q_eig = np.percentile(mu_plug_q_eig,  2.5, axis=2)
hi_q_eig = np.percentile(mu_plug_q_eig, 97.5, axis=2)

plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(grid, mu_str_q_eig_mean[q], color=colors[q], lw=2,
             label=f"Structural {quartile_labels[q]}")
    plt.plot(grid, mu_plug_q_eig_mean[q], "--", color=colors[q], lw=2)
    plt.fill_between(grid, lo_q_eig[q], hi_q_eig[q], color=colors[q], alpha=.15)
plt.title("Eigen-normalised DRF by Degree Quartile")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\mu_q(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
#  GRADIENT OF THE DOSE–RESPONSE  (∂/∂t μ)
# ================================================================

t_vals = grid        # rename for readability

# -----------------------------------------------------------------
# 1.  GLOBAL GRADIENT  (row-normalised)
# -----------------------------------------------------------------
grad_plug_row = np.gradient(mu_plug_row, t_vals, axis=1)   # n_sim × g
grad_str_row  = np.gradient(mu_str_row,  t_vals, axis=1)
grad_plug_row_mean, lo_grad_row, hi_grad_row = mean_lo_hi(grad_plug_row)
grad_str_row_mean, *_ = mean_lo_hi(grad_str_row)

plt.figure(figsize=(10, 5))
plt.plot(t_vals, grad_str_row_mean, lw=2, label="Structural ∂μ/∂t")
plt.plot(t_vals, grad_plug_row_mean, "--", lw=2, label="Net-GPS ∂μ/∂t")
plt.fill_between(t_vals, lo_grad_row, hi_grad_row, alpha=.15, colour='orange')
plt.title("Global Gradient – Row-normalised W")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\partial_t \mu(t)$")
plt.grid(True), plt.legend(), plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# 2.  QUARTILE-SPECIFIC GRADIENTS
# -----------------------------------------------------------------
# --- compute gradients (4 × g × n_sim) ---
grad_plug_q_row  = np.gradient(mu_plug_q_row,  t_vals, axis=1)
grad_str_q_row   = np.gradient(mu_str_q_row,   t_vals, axis=1)
grad_plug_q_eig  = np.gradient(mu_plug_q_eig,  t_vals, axis=1)
grad_str_q_eig   = np.gradient(mu_str_q_eig,   t_vals, axis=1)

# --- average over simulations (axis=2) ---
grad_plug_q_row_mean = grad_plug_q_row.mean(axis=2)
grad_str_q_row_mean  = grad_str_q_row.mean(axis=2)
grad_plug_q_eig_mean = grad_plug_q_eig.mean(axis=2)
grad_str_q_eig_mean  = grad_str_q_eig.mean(axis=2)

# -------- plot: ROW-normalised -----------------------------------
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals, grad_str_q_row_mean[q],
             lw=2, color=colors[q],
             label=f"Structural {quartile_labels[q]}")
    plt.plot(t_vals, grad_plug_q_row_mean[q],
             "--", lw=2, color=colors[q])
plt.title("Gradient by Degree Quartile (Row-normalised)")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\partial_t \mu_q(t)$")
plt.grid(True), plt.legend(), plt.tight_layout()
plt.show()

# -------- plot: EIGEN-normalised ---------------------------------
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals, grad_str_q_eig_mean[q],
             lw=2, color=colors[q],
             label=f"Structural {quartile_labels[q]}")
    plt.plot(t_vals, grad_plug_q_eig_mean[q],
             "--", lw=2, color=colors[q])
plt.title("Gradient by Degree Quartile (Eigen-normalised)")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\partial_t \mu_q(t)$")
plt.grid(True), plt.legend(), plt.tight_layout()
plt.show()

