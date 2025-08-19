import numpy as np
import networkx as nx
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import random


# --- Configuration ---
config = {
    "n": 500,
    "p_edge": 0.1,
    "rho": 0.7,
    "beta": 1,
    "sigma_T": 2.0,
    "n_grid": 50,
    "n_outer": 1,    # number of network draws
    "n_inner": 500,     # number of noise draws per network
    "alpha" : 2,
    "sigma_U" : 1,
    "sigma_H" : 3,
    "alpha_T" : 2,
    "second_stage_terms": ["WY_hat", "T", "R", "T**2", "R**2",  "T*R"],
    "dgp_terms" : ["T","R", "T*R"],
    "dgp_coeffs": {"T": 3, "R": 2,"T*R":15},
    #"dgp_terms" : ["T","R","T**2","R**2","T*R"],
    #"dgp_coeffs" : {"T": 3},
    "norm_method" : "row",
    "graph" : 'scalefree',
    'm_attach': 10,
    'seed':202507
}

random.seed(config['seed'])
np.random.seed(config["seed"])
# --- Grid of t values ---
t_vals = np.linspace(config["alpha_T"] - 3 * np.sqrt(config["beta"]**2 * config["sigma_H"]**2 + config["sigma_T"]**2), config["alpha_T"] + 3 * np.sqrt(config["beta"]**2 * config["sigma_H"]**2 + config["sigma_T"]**2), config["n_grid"])


# --- Store results ---
mu_structural_all_row = []
mu_plugin_all_row = []
p_value_all_row = []

mu_structural_all_eigen = []
mu_plugin_all_eigen = []
p_value_all_eigen = []

mu_struct_q_row    = np.zeros((4, config["n_grid"], config["n_inner"]))
mu_plugin_q_row    = np.zeros_like(mu_struct_q_row)
mu_struct_q_eigen  = np.zeros_like(mu_struct_q_row)
mu_plugin_q_eigen  = np.zeros_like(mu_struct_q_row)
mu_naive_row      = np.zeros((config["n_inner"], config["n_grid"]))
mu_naive_eigen    = np.zeros_like(mu_naive_row)          # fit separately to Y_eigen

df_params_row = pd.DataFrame(columns=["intercept"] + config["second_stage_terms"], index=range(config["n_inner"]))
df_params_eigen = pd.DataFrame(columns=["intercept"] + config["second_stage_terms"], index=range(config["n_inner"]))
df_params_row_naive = pd.DataFrame(columns=["intercept", "T", "T**2"], index=range(config["n_inner"]))
df_params_eigen_naive = pd.DataFrame(columns=["intercept", "T", "T**2"], index=range(config["n_inner"]))


def build_design_matrix(terms, variables):
    X_list = []
    for term in terms:
            local_vars = {k: variables[k] for k in variables}
            X_list.append(eval(term, {"np": np}, local_vars))
    return np.column_stack(X_list)

def eval_terms(terms, coeffs, variables):
    res = np.zeros_like(next(iter(variables.values())))
    for term in terms:
        res += coeffs[term] * eval(term, {"np": np, "sin": np.sin, "exp": np.exp, "log": np.log, "cos": np.cos, "tanh": np.tanh, "sqrt": np.sqrt}, variables)
    return res

def normalize(W, method="row"):
    if method == "row":
        W_norm = W / W.sum(axis=1, keepdims=True)
    elif method == "eigen":
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        W_norm = W / eig_max
    else:
        raise ValueError("Unsupported normalization method.")
    return np.nan_to_num(W_norm)

for outer in range(config["n_outer"]):
    # --- Generate Network ---
    
    if config['graph'] == 'erdos_renyi':
        G = nx.erdos_renyi_graph(config["n"], config["p_edge"])
    elif config['graph'] == 'scalefree':
        G = nx.barabasi_albert_graph(config["n"], config['m_attach'])

    G_array = nx.to_numpy_array(G)
    #W = normalize(G_array, method = config['norm_method'])
    W_row = normalize(G_array, method = 'row')
    W_eigen = normalize(G_array, method = 'eigen')
    pd.DataFrame(W_row).to_csv('C:/Users/psfin/Documents/W_row_sf.csv',index=False) # save DF to ensure it's the same across all simulation settings
    pd.DataFrame(W_eigen).to_csv('C:/Users/psfin/Documents/W_eigen_sf.csv', index=False)
    # W_eigen = pd.read_csv('C:/Users/psfin/Documents/W_eigen_er.csv').to_numpy()
    # W_row = pd.read_csv('C:/Users/psfin/Documents/W_row_er.csv').to_numpy()

    degrees = np.asarray([d for _, d in G.degree()])
    quantiles = np.quantile(degrees, [0, .25, .50, .75, 1.0])
    deg_bin = np.digitize(degrees, quantiles[1:-1])   # 0,1,2,3

    #W[np.isnan(W)] = 0
    I = np.eye(config["n"])
    inv_matrix_row = np.linalg.inv(I - config["rho"] * W_row)
    inv_matrix_eigen = np.linalg.inv(I - config["rho"] * W_eigen)

    # --- Generate Base Covariates and Treatments ---
    H = np.random.normal(0, config['sigma_H'], (config["n"], 1))
    T = config['alpha_T'] + config["beta"] * H + np.random.normal(0, config["sigma_T"], (config["n"], 1))
    
    # non-parametric:
    reg = RandomForestRegressor().fit(H, T.flatten())
    T_hat = reg.predict(H).reshape(-1, 1)

    sigma_T_hat = np.std(T - T_hat)
    R_obs = norm.pdf(T, loc=T_hat, scale=sigma_T_hat)
    TX_obs = H * R_obs

    if outer == 0:
        t_vals = np.linspace(math.floor(np.min(T)), math.ceil(np.max(T)), config['n_grid'])


    # --- Precompute Instruments (W^x H) --
    W_rowX  = W_row @ H
    W_row2X = W_row @ W_rowX
    W_row3X = W_row @ W_row2X
    W_row4X = W_row @ W_row3X
    W_rowR  = W_row @ R_obs
    W_row2R = W_row @ W_rowR
    W_row3R = W_row @ W_row2R
    W_row4R = W_row @ W_row3R
    W_rowTX = W_row @ TX_obs

    W_eigenX  = W_eigen @ H
    W_eigen2X = W_eigen @ W_eigenX
    W_eigen3X = W_eigen @ W_eigen2X
    W_eigen4X = W_eigen @ W_eigen3X
    W_eigenR  = W_eigen @ R_obs
    W_eigen2R = W_eigen @ W_eigenR
    W_eigen3R = W_eigen @ W_eigen2R
    W_eigen4R = W_eigen @ W_eigen3R
    W_eigenTX = W_eigen @ TX_obs


    X_instr_row = np.hstack([W_rowX, W_row2X, W_row3X, W_row4X, W_rowR, W_row2R, W_row3R, W_row4R, W_rowTX])
    Z_full_row = sm.add_constant(X_instr_row)

    X_instr_eigen = np.hstack([W_eigenX, W_eigen2X, W_eigen3X, W_eigen4X, W_eigenR, W_eigen2R, W_eigen3R, W_eigen4R, W_eigenTX])
    Z_full_eigen = sm.add_constant(X_instr_eigen)


    # --- Initialize useful vectors ---
    mu_structural_row = np.zeros((config["n_grid"], config["n_inner"]))
    mu_plugin_row = np.zeros((config["n_grid"], config["n_inner"]))
    bias_row = np.zeros((config["n_grid"], config["n_inner"]))

    mu_structural_eigen = np.zeros((config["n_grid"], config["n_inner"]))
    mu_plugin_eigen = np.zeros((config["n_grid"], config["n_inner"]))
    bias_eigen = np.zeros((config["n_grid"], config["n_inner"]))



    # --- Run a noise simulation ---
    for inner in range(config["n_inner"]):
        print(inner)
        u_fixed = np.random.normal(0, config['sigma_U'], (config["n"], 1))
        
        #Y_C = inv_matrix @ (config['alpha'] + config['beta_T'] * T + config['beta_R'] * R_obs + config["beta_TR"] * (T * R_obs) + u_fixed)
        variables = {"T": T, "R": R_obs}
        eta = eval_terms(config["dgp_terms"], config["dgp_coeffs"], variables)
        Y_row = inv_matrix_row @ (config["alpha"] + eta + u_fixed)
        Y_eigen = inv_matrix_eigen @ (config["alpha"] + eta + u_fixed)

        WY_target_row = (W_row @ Y_row).flatten()
        WY_target_eigen = (W_eigen @ Y_eigen).flatten()
        instrument_model_row = sm.OLS(WY_target_row.ravel(), Z_full_row).fit()
        instrument_model_eigen = sm.OLS(WY_target_eigen.ravel(), Z_full_eigen).fit()
        WY_hat_row = instrument_model_row.predict(Z_full_row).reshape(-1, 1)
        WY_hat_eigen = instrument_model_eigen.predict(Z_full_eigen).reshape(-1, 1)

        variables_row = {"T": T, "R": R_obs,  "WY_hat": WY_hat_row}
        variables_eigen = {"T": T, "R": R_obs,  "WY_hat": WY_hat_eigen}

        X_stage2_row = build_design_matrix(config["second_stage_terms"], variables_row)
        X_stage2_row = sm.add_constant(X_stage2_row, has_constant='add')

        X_stage2_eigen = build_design_matrix(config["second_stage_terms"], variables_eigen)
        X_stage2_eigen = sm.add_constant(X_stage2_eigen, has_constant='add')

        #X_stage2 = sm.add_constant(np.hstack([WY_hat, T, R_obs, T**2, R_obs**2, T * R_obs]))
        model_row = sm.OLS(Y_row.ravel(), X_stage2_row).fit()
        model_eigen = sm.OLS(Y_eigen.ravel(), X_stage2_eigen).fit()

        beta_hat_row = model_row.params
        beta_hat_eigen = model_eigen.params

        beta_hat_reduced_row = [x for j, x in enumerate(beta_hat_row) if j != 1]
        est_inv_row = np.linalg.inv(np.eye(config['n']) - beta_hat_row[1]*W_row)

        beta_hat_reduced_eigen = [x for j, x in enumerate(beta_hat_eigen) if j != 1]
        est_inv_eigen = np.linalg.inv(np.eye(config['n']) - beta_hat_eigen[1]*W_eigen)

        X_naive = build_design_matrix(["T", "T**2","T**3"], {"T": T})
        X_naive = sm.add_constant(X_naive, has_constant="add")

        mod_naive_row   = sm.OLS(Y_row.ravel(),   X_naive).fit()
        mod_naive_eigen = sm.OLS(Y_eigen.ravel(), X_naive).fit()

        for t_idx, t_val in enumerate(t_vals):

            X_pred_naive = build_design_matrix(["T", "T**2","T**3"], {"T": t_val})
            X_pred_naive = sm.add_constant(X_pred_naive, has_constant="add")

            mu_n_row   = (X_pred_naive @ mod_naive_row.params).mean()
            mu_n_eigen = (X_pred_naive @ mod_naive_eigen.params).mean()

            mu_naive_row[inner, t_idx]   = mu_n_row
            mu_naive_eigen[inner, t_idx] = mu_n_eigen

            Y_struct_list_row = []
            Y_plugin_list_row = []

            Y_struct_list_eigen = []
            Y_plugin_list_eigen = []


            for i in range(config["n"]):
                # Structural
                T_local = T.copy()
                T_local[i] = t_val
                R_local = norm.pdf(T_local, loc=T_hat, scale=sigma_T_hat)
                TR_local = T_local * R_local

                variables_local = {"T": T_local, "R": R_local}
                eta_local = eval_terms(config["dgp_terms"], config["dgp_coeffs"], variables_local)
                Y_local_row = inv_matrix_row @ (config["alpha"] + eta_local + u_fixed)
                Y_local_eigen = inv_matrix_eigen @ (config["alpha"] + eta_local + u_fixed)


                # Y_local_c = inv_matrix @ (config['alpha'] + config['beta_T'] * T_local + config['beta_R']* R_local + config['beta_TR'] * TR_local + u_fixed)
                Y_struct_list_row.append(Y_local_row[i])
                Y_struct_list_eigen.append(Y_local_eigen[i])


                # Network-GPS (Net-GPS Estimator)
                terms_all = config["second_stage_terms"]
                terms_pred = [term for term in terms_all if "WY_hat" not in term]
                variables_pred = {"T": T_local, "R": R_local}
                X_pred = build_design_matrix(terms_pred, variables_pred)
                X_pred = sm.add_constant(X_pred, has_constant='add')

                #X_pred_c = np.hstack([np.ones((config["n"], 1)), T_local, R_local, T_local**2, R_local**2,  T_local * R_local])
                rhs_row = X_pred @ beta_hat_reduced_row
                rhs_eigen = X_pred @ beta_hat_reduced_eigen

                res_row = est_inv_row @ rhs_row
                res_eigen = est_inv_eigen @ rhs_eigen

                Y_netgps_row = res_row[i]
                Y_plugin_list_row.append(Y_netgps_row)
                
                Y_netgps_eigen = res_eigen[i]
                Y_plugin_list_eigen.append(Y_netgps_eigen)

                #print(i)

            # --- turn node-level lists into flat arrays -----------------
            Y_struct_row  = np.asarray(Y_struct_list_row).ravel()
            Y_plugin_row  = np.asarray(Y_plugin_list_row).ravel()
            Y_struct_eig  = np.asarray(Y_struct_list_eigen).ravel()
            Y_plugin_eig  = np.asarray(Y_plugin_list_eigen).ravel()

            # --- degree-quartile means ----------------------------------
            for b in range(4):
                mask = deg_bin == b            # deg_bin was defined once per network
                mu_struct_q_row[b, t_idx, inner]   = Y_struct_row[mask].mean()
                mu_plugin_q_row[b, t_idx, inner]   = Y_plugin_row[mask].mean()
                mu_struct_q_eigen[b, t_idx, inner] = Y_struct_eig[mask].mean()
                mu_plugin_q_eigen[b, t_idx, inner] = Y_plugin_eig[mask].mean()

            # --- overall (all-nodes) means — uses the same arrays --------
            mu_structural_row[t_idx,   inner] = Y_struct_row.mean()
            mu_plugin_row[t_idx,       inner] = Y_plugin_row.mean()
            mu_structural_eigen[t_idx, inner] = Y_struct_eig.mean()
            mu_plugin_eigen[t_idx,     inner] = Y_plugin_eig.mean()

        bias_row = mu_plugin_row - mu_structural_row
        bias_eigen = mu_plugin_eigen - mu_structural_eigen

        t_stat_row, p_value_row = stats.ttest_1samp(bias_row, popmean=0.0, axis=1)
        p_value_all_row.append(p_value_row)

        t_stat_eigen, p_value_eigen = stats.ttest_1samp(bias_eigen, popmean=0.0, axis=1)
        p_value_all_eigen.append(p_value_eigen)


    mu_structural_all_row.append(mu_structural_row.mean(axis=1))
    mu_plugin_all_row.append(mu_plugin_row.mean(axis=1))

    mu_structural_all_eigen.append(mu_structural_eigen.mean(axis=1))
    mu_plugin_all_eigen.append(mu_plugin_eigen.mean(axis=1))



# --- Aggregate results ---
mu_structural_all_row = np.array(mu_structural_all_row)  # shape: (n_outer, n_grid)
mu_plugin_all_row = np.array(mu_plugin_all_row)

mu_structural_all_eigen = np.array(mu_structural_all_eigen)  # shape: (n_outer, n_grid)
mu_plugin_all_eigen = np.array(mu_plugin_all_eigen)

np.asarray(mu_plugin_all_row) - np.asarray(mu_structural_all_row)
np.asarray(mu_plugin_all_eigen) - np.asarray(mu_structural_all_eigen)

from sklearn.neighbors import KernelDensity

mu_structural_row; mu_plugin_row
mu_structural_eigen; mu_plugin_eigen

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(T)
log_dens = kde.score_samples(t_vals.reshape(-1, 1))
dens = np.exp(log_dens)
weights = dens / dens.sum()
bias = np.asarray(mu_plugin_all_row).mean(axis=0) - np.array(mu_structural_all_row).mean(axis=0)
weighted_mean_bias = np.sum(weights * bias)                    # ∑ w * bias
weighted_rmse = np.sqrt(np.sum(weights * bias**2))

err_row   = mu_plugin_row.T - mu_structural_row.T       # plugin – structural
err_eigen = mu_plugin_eigen.T - mu_structural_eigen.T

bias_g_row = err_row.mean(axis=0)                 # Bias(t_g)
mse_g_row  = (err_row**2).mean(axis=0)            # MSE(t_g)

weighted_bias_row = np.dot(bias_g_row, weights)
weighted_rmse_row = np.sqrt(np.dot(mse_g_row, weights))

# ---- EIGEN normalisation ----------------------------------------
bias_g_eig = err_eigen.mean(axis=0)
mse_g_eig  = (err_eigen**2).mean(axis=0)

weighted_bias_eig = np.dot(bias_g_eig, weights)
weighted_rmse_eig = np.sqrt(np.dot(mse_g_eig, weights))
############## CONFIDENCE INTERVALS

def weighted_bias_rmse(err, weights):
    bias_g = err.mean(axis=0)          # axis=0 because sims are rows
    mse_g  = (err**2).mean(axis=0)
    wbias  = np.dot(bias_g, weights)
    wrmse  = np.sqrt(np.dot(mse_g, weights))
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
    idx = np.random.randint(0, G, size=G)           # resample columns
    err_boot = err_row[idx, :]                      # same shape (G, S)
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
err_naive_row   = mu_naive_row   - mu_structural_row.T
err_naive_eigen = mu_naive_eigen - mu_structural_eigen.T

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
grad_true_row = np.gradient(mu_structural_row, axis=0)  # shape (n_sim, n_grid)
grad_est_row  = np.gradient(mu_plugin_row,  axis=0)
grad_naive_row = np.gradient(mu_naive_row.T, axis=0)

grad_true_eigen = np.gradient(mu_structural_eigen, axis=0)  # shape (n_sim, n_grid)
grad_est_eigen  = np.gradient(mu_plugin_eigen,  axis=0)
grad_naive_eigen = np.gradient(mu_naive_eigen.T, axis=0)


# Compute gradient bias matrix
grad_bias_row = grad_est_row - grad_true_row  # shape (n_sim, n_grid)
grad_bias_naive_row = grad_est_row - grad_naive_row

grad_bias_eigen = grad_est_eigen - grad_true_eigen  # shape (n_sim, n_grid)
grad_bias_naive_eigen = grad_est_eigen - grad_naive_eigen


# Average across sims and grid
mean_grad_bias_row, rmse_grad_bias_row = weighted_bias_rmse(grad_bias_row.T, weights)
mean_grad_bias_naive_row, rmse_grad_bias_naive_row = weighted_bias_rmse(grad_bias_naive_row.T, weights)

mean_grad_bias_eigen, rmse_grad_bias_eigen = weighted_bias_rmse(grad_bias_eigen.T, weights)
mean_grad_bias_naive_eigen, rmse_grad_bias_naive_eigen = weighted_bias_rmse(grad_bias_naive_eigen.T, weights)

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

grad_ci_bias_row, grad_ci_rmse_row = get_grad_ci(grad_bias_row.T)
grad_ci_bias_naive_row, grad_ci_rmse_naive_row = get_grad_ci(grad_bias_naive_row.T)
grad_ci_bias_eigen, grad_ci_rmse_eigen = get_grad_ci(grad_bias_eigen.T)
grad_ci_bias_naive_eigen, grad_ci_rmse_naive_eigen = get_grad_ci(grad_bias_naive_eigen.T)

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
summary.to_csv('C:/Users/psfin/Documents/results_local_'+config["graph"]+'_'+config['norm_method']+'_new.csv',index=False)



################
# --- Use only last draw for DRF plots ---
mu_structural_last_mean_row = mu_structural_row.mean(axis=1)
mu_plugin_last_mean_row = mu_plugin_row.mean(axis=1)

mu_structural_last_mean_eigen = mu_structural_eigen.mean(axis=1)
mu_plugin_last_mean_eigen = mu_plugin_eigen.mean(axis=1)

netgps_lo_row = np.percentile(mu_plugin_row, 2.5, axis=1)
netgps_hi_row = np.percentile(mu_plugin_row, 97.5, axis=1)

netgps_lo_eigen = np.percentile(mu_plugin_eigen, 2.5, axis=1)
netgps_hi_eigen = np.percentile(mu_plugin_eigen, 97.5, axis=1)

grad_structural_row = np.gradient(mu_structural_last_mean_row, t_vals)
grad_plugin_row = np.gradient(mu_plugin_last_mean_row, t_vals)

grad_structural_eigen = np.gradient(mu_structural_last_mean_eigen, t_vals)
grad_plugin_eigen = np.gradient(mu_plugin_last_mean_eigen, t_vals)

gradients_last_row = np.gradient(mu_plugin_row, t_vals, axis=0)  # shape: (n_inner, n_grid)
grad_last_lo_row = np.percentile(gradients_last_row, 2.5, axis=1)
grad_last_hi_row = np.percentile(gradients_last_row, 97.5, axis=1)

gradients_last_eigen = np.gradient(mu_plugin_eigen, t_vals, axis=0)  # shape: (n_inner, n_grid)
grad_last_lo_eigen = np.percentile(gradients_last_eigen, 2.5, axis=1)
grad_last_hi_eigen = np.percentile(gradients_last_eigen, 97.5, axis=1)


# --- Use full set for bias plots ---
mu_structural_mean_row = mu_structural_all_row.mean(axis=0)
mu_plugin_mean_row = mu_plugin_all_row.mean(axis=0)
bias_mean_row = mu_plugin_mean_row - mu_structural_mean_row
bias_netgps_all_row = mu_plugin_all_row - mu_structural_mean_row

mu_structural_mean_eigen = mu_structural_all_eigen.mean(axis=0)
mu_plugin_mean_eigen = mu_plugin_all_eigen.mean(axis=0)
bias_mean_eigen = mu_plugin_mean_eigen - mu_structural_mean_eigen
bias_netgps_all_eigen = mu_plugin_all_eigen - mu_structural_mean_eigen

bias_netgps_lo_row = np.percentile(bias_netgps_all_row, 2.5, axis=0)
bias_netgps_hi_row = np.percentile(bias_netgps_all_row, 97.5, axis=0)

bias_netgps_lo_eigen = np.percentile(bias_netgps_all_eigen, 2.5, axis=0)
bias_netgps_hi_eigen = np.percentile(bias_netgps_all_eigen, 97.5, axis=0)


# --- Gradient Bias from all draws ---
grad_plugin_all_row = np.gradient(mu_plugin_all_row, t_vals, axis=1)
grad_structural_all_row = np.gradient(mu_structural_all_row, t_vals, axis=1)
grad_bias_all_row = grad_plugin_all_row - grad_structural_all_row

grad_plugin_all_eigen = np.gradient(mu_plugin_all_eigen, t_vals, axis=1)
grad_structural_all_eigen = np.gradient(mu_structural_all_eigen, t_vals, axis=1)
grad_bias_all_eigen = grad_plugin_all_eigen - grad_structural_all_eigen


grad_bias_row = grad_bias_all_row.mean(axis=0)
grad_bias_lo_row = np.percentile(grad_bias_all_row, 2.5, axis=0)
grad_bias_hi_row = np.percentile(grad_bias_all_row, 97.5, axis=0)

grad_bias_eigen = grad_bias_all_eigen.mean(axis=0)
grad_bias_lo_eigen = np.percentile(grad_bias_all_eigen, 2.5, axis=0)
grad_bias_hi_eigen = np.percentile(grad_bias_all_eigen, 97.5, axis=0)


# --- Plot: DRF (Single Network Draw) ---
plt.figure(figsize=(10, 5))
plt.plot(t_vals, mu_structural_last_mean_row, label="Structural Local DRF Row-normalized", linewidth=2, color="blue")
plt.plot(t_vals, mu_plugin_last_mean_row, label="Net-GPS Estimate Row-normalized", linestyle="--", linewidth=2, color="blue")
plt.plot(t_vals, mu_structural_last_mean_eigen, label="Structural Local DRF Eigenvalue-normalized", linewidth=2, color="orange")
plt.plot(t_vals, mu_plugin_last_mean_eigen, label="Net-GPS Estimate Eigenvalue-nrormalized", linestyle="--", linewidth=2, color="orange")
plt.fill_between(t_vals, netgps_lo_row, netgps_hi_row, color="blue", alpha=0.2, label="Estimator 95% CI Row-normalized")
plt.fill_between(t_vals, netgps_lo_eigen, netgps_hi_eigen, color="orange", alpha=0.2, label="Estimator 95% CI Eigenvalue-normalized")

plt.xlabel("Treatment Level $t$")
plt.ylabel("Expected Outcome $\\mu(t)$")
plt.title("Local Dose-Response Function — Single Network Draw")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot: Bias (Across All Network Draws) ---
plt.figure(figsize=(10, 5))
plt.plot(t_vals, bias_mean_row, label="Bias: Net-GPS - Structural (avg. over draws)", linewidth=2, color="blue")
plt.fill_between(t_vals, bias_netgps_lo_row, bias_netgps_hi_row, color="blue", alpha=0.2, label="Bias 95% CI Row-normalized")
plt.plot(t_vals, bias_mean_eigen, label="Bias: Net-GPS - Structural (avg. over draws)", linewidth=2, color="orange")
plt.fill_between(t_vals, bias_netgps_lo_eigen, bias_netgps_hi_eigen, color="orange", alpha=0.2, label="Bias 95% CI Eigenvalue-normalized")


plt.xlabel("Treatment Level $t$")
plt.ylabel("Bias")
plt.title("Bias of Net-GPS Local DRF Estimate — Averaged over Simulations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot: Gradient (Single Network Draw) ---
plt.figure(figsize=(10, 5))
plt.plot(t_vals, grad_structural_row, label="Structural Gradient Row", linewidth=2, color="blue")
plt.plot(t_vals, grad_plugin_row, label="Net-GPS Gradient Row", linestyle="--", linewidth=2, color="blue")
plt.fill_between(t_vals, grad_last_lo_row, grad_last_hi_row, color="blue", alpha=0.2, label="Gradient 95% CI Row")

plt.plot(t_vals, grad_structural_eigen, label="Structural Gradient Eigen", linewidth=2, color="orange")
plt.plot(t_vals, grad_plugin_eigen, label="Net-GPS Gradient Eigen", linestyle="--", linewidth=2, color="orange")
plt.fill_between(t_vals, grad_last_lo_eigen, grad_last_hi_eigen, color="orange", alpha=0.2, label="Gradient 95% CI Eigen")


plt.xlabel("Treatment Level $t$")
plt.ylabel("Gradient $\\frac{d\\mu(t)}{dt}$")
plt.title("Gradient of Local Dose-Response Function — Single Network Draw")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot: Gradient Bias (Across All Network Draws) ---
plt.figure(figsize=(10, 5))
plt.plot(t_vals, grad_bias_row, label="Gradient Bias Row", linewidth=2, color="blue")
plt.fill_between(t_vals, grad_bias_lo_row, grad_bias_hi_row, color="blue", alpha=0.2, label="Gradient Bias 95% CI Row")

plt.plot(t_vals, grad_bias_eigen, label="Gradient Bias Eigen", linewidth=2, color="orange")
plt.fill_between(t_vals, grad_bias_lo_eigen, grad_bias_hi_eigen, color="orange", alpha=0.2, label="Gradient Bias 95% CI Eigen")


plt.xlabel("Treatment Level $t$")
plt.ylabel("Gradient Bias")
plt.title("Bias in Gradient of Net-GPS Estimate — Averaged over Simulations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================================================================
#  DEGREE-QUARTILE SUMMARY & PLOTS 
# ================================================================

# ---- 1. Collapse over the inner draws ---------------------------------
mu_struct_q_row_mean   = mu_struct_q_row.mean(axis=2)     # (4, n_grid)
mu_plugin_q_row_mean   = mu_plugin_q_row.mean(axis=2)
mu_struct_q_eig_mean   = mu_struct_q_eigen.mean(axis=2)
mu_plugin_q_eig_mean   = mu_plugin_q_eigen.mean(axis=2)

lo_q_row = np.percentile(mu_plugin_q_row,    2.5, axis=2)
hi_q_row = np.percentile(mu_plugin_q_row,   97.5, axis=2)
lo_q_eig = np.percentile(mu_plugin_q_eigen,  2.5, axis=2)
hi_q_eig = np.percentile(mu_plugin_q_eigen, 97.5, axis=2)

# Bias per quartile
bias_q_row = mu_plugin_q_row_mean - mu_struct_q_row_mean
bias_q_eig = mu_plugin_q_eig_mean - mu_struct_q_eig_mean

bias_draws_row = mu_plugin_q_row - mu_struct_q_row        # (4, n_grid, n_inner)
bias_lo_q_row  = np.percentile(bias_draws_row,  2.5, axis=2)
bias_hi_q_row  = np.percentile(bias_draws_row, 97.5, axis=2)

bias_draws_eig = mu_plugin_q_eigen - mu_struct_q_eigen
bias_lo_q_eig  = np.percentile(bias_draws_eig,  2.5, axis=2)
bias_hi_q_eig  = np.percentile(bias_draws_eig, 97.5, axis=2)

quartile_labels = ["Q1: lowest deg.", "Q2", "Q3", "Q4: hubs"]
colors = ["steelblue", "forestgreen", "goldenrod", "firebrick"]

# ---- 2. DRF by quartile (ROW-normalised) -------------------------------
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals,
             mu_struct_q_row_mean[q],
             lw=2, color=colors[q],
             label=f"Structural {quartile_labels[q]}")
    plt.plot(t_vals,
             mu_plugin_q_row_mean[q],
             ls="--", lw=2, color=colors[q])
    plt.fill_between(t_vals, lo_q_row[q], hi_q_row[q],
                     alpha=.15, color=colors[q])
plt.title("Row-normalised DRF by Degree Quartile")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\mu_q(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---- 3. DRF by quartile (EIGEN-normalised) ----------------------------
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals,
             mu_struct_q_eig_mean[q],
             lw=2, color=colors[q],
             label=f"Structural {quartile_labels[q]}")
    plt.plot(t_vals,
             mu_plugin_q_eig_mean[q],
             ls="--", lw=2, color=colors[q])
    plt.fill_between(t_vals, lo_q_eig[q], hi_q_eig[q],
                     alpha=.15, color=colors[q])
plt.title("Eigen-normalised DRF by Degree Quartile")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\mu_q(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---- 4. Bias by quartile (ROW) ----------------------------------------
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals, bias_q_row[q],
             lw=2, color=colors[q],
             label=quartile_labels[q])
    plt.fill_between(t_vals, bias_lo_q_row[q], bias_hi_q_row[q],
                     alpha=.15, color=colors[q])
plt.title("Bias: Net-GPS − Structural (Row-normalised)")
plt.xlabel("Treatment level $t$")
plt.ylabel("Bias")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---- 5. Bias by quartile (EIGEN) --------------------------------------
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals, bias_q_eig[q],
             lw=2, color=colors[q],
             label=quartile_labels[q])
    plt.fill_between(t_vals, bias_lo_q_eig[q], bias_hi_q_eig[q],
                     alpha=.15, color=colors[q])
plt.title("Bias: Net-GPS − Structural (Eigen-normalised)")
plt.xlabel("Treatment level $t$")
plt.ylabel("Bias")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

##### Gradient per quartile

# Compute gradients of plugin and structural estimates per quartile
grad_plugin_q_row = np.gradient(mu_plugin_q_row, t_vals, axis=1)
grad_struct_q_row = np.gradient(mu_struct_q_row, t_vals, axis=1)

grad_plugin_q_eig = np.gradient(mu_plugin_q_eigen, t_vals, axis=1)
grad_struct_q_eig = np.gradient(mu_struct_q_eigen, t_vals, axis=1)

# Average over inner draws
grad_plugin_q_row_mean = grad_plugin_q_row.mean(axis=2)
grad_struct_q_row_mean = grad_struct_q_row.mean(axis=2)
grad_bias_q_row = grad_plugin_q_row_mean - grad_struct_q_row_mean

grad_plugin_q_eig_mean = grad_plugin_q_eig.mean(axis=2)
grad_struct_q_eig_mean = grad_struct_q_eig.mean(axis=2)
grad_bias_q_eig = grad_plugin_q_eig_mean - grad_struct_q_eig_mean

# --- Plot: Gradient by Degree Quartile (ROW-normalised) ---
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals,
             grad_struct_q_row_mean[q],
             lw=2, color=colors[q],
             label=f"Structural {quartile_labels[q]}")
    plt.plot(t_vals,
             grad_plugin_q_row_mean[q],
             ls="--", lw=2, color=colors[q])
plt.title("Gradient by Degree Quartile (Row-normalised)")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\partial_t \mu_q(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot: Gradient by Degree Quartile (EIGEN-normalised) ---
plt.figure(figsize=(10, 6))
for q in range(4):
    plt.plot(t_vals,
             grad_struct_q_eig_mean[q],
             lw=2, color=colors[q],
             label=f"Structural {quartile_labels[q]}")
    plt.plot(t_vals,
             grad_plugin_q_eig_mean[q],
             ls="--", lw=2, color=colors[q])
plt.title("Gradient by Degree Quartile (Eigen-normalised)")
plt.xlabel("Treatment level $t$")
plt.ylabel(r"$\partial_t \mu_q(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(degrees, bins=30, alpha=0.6, label='Original Degree Distribution')
plt.title('Node Degree Distribution (Barabási-Albert)')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(W_row.sum(axis=1), bins=30, alpha=0.6, label='Row-normalized Degree')
plt.hist(W_eigen.sum(axis=1), bins=30, alpha=0.6, label='Eigen-normalized Degree')
plt.title('Sum of Normalized Adjacency Rows')
plt.xlabel('Row Sum (Influence Weight Total)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(W_row, cmap="viridis", aspect='auto')
plt.title("Row-normalized W")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(W_eigen, cmap="viridis", aspect='auto')
plt.title("Eigenvalue-normalized W")
plt.colorbar()
plt.tight_layout()
plt.show()