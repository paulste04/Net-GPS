## Simulation 1: Violating ignorability
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set seed
np.random.seed(42)

# Parameters
n = 500
n_sim = 100
true_tau = 2.0
beta = 1.0

# Store estimates
results = []

for sim in range(n_sim):
    # Simulate data
    X = np.random.normal(0, 1, n)
    T = 0.5 * X + np.random.normal(0, 1, n)
    Y = true_tau * T + beta * X + np.random.normal(0, 1, n)

    # Naive OLS: Y ~ T
    model_naive = sm.OLS(Y, sm.add_constant(T)).fit()
    tau_naive = model_naive.params[1]

    # Correct OLS: Y ~ T + X
    model_correct = sm.OLS(Y, sm.add_constant(np.column_stack([T, X]))).fit()
    tau_correct = model_correct.params[1]

    # Store results
    results.append({
        "Simulation": sim,
        "Estimator": "Naive (violates ignorability)",
        "Tau_hat": tau_naive,
        "Bias": tau_naive - true_tau
    })
    results.append({
        "Simulation": sim,
        "Estimator": "Correct (satisfies ignorability)",
        "Tau_hat": tau_correct,
        "Bias": tau_correct - true_tau
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Plot 1: KDE of estimated tau
plt.figure(figsize=(6, 5))
sns.kdeplot(data=df_results, x="Tau_hat", hue="Estimator", common_norm=False, alpha=0.7)
plt.axvline(true_tau, color="black", linestyle="--", label="True τ = 2.0")
plt.title("Sampling Distribution of Estimated Treatment Effect")
plt.xlabel("Estimated Treatment Effect (τ̂)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()


# Plot 2: Bias strip plot
plt.figure(figsize=(6, 5))
sns.stripplot(data=df_results, x="Estimator", y="Bias", jitter=True, alpha=0.6)
sns.pointplot(data=df_results, x="Estimator", y="Bias", join=False, color='red', markers='D', ci='sd', errwidth=1.5)
plt.axhline(0, color="black", linestyle="--")
plt.title("Bias in Estimated Treatment Effect")
plt.ylabel("Bias (τ̂ - τ)")
plt.tight_layout()
plt.show()

###### Simulation 2: Violating SUTVA

import networkx as nx

n = 100
n_sim = 50
true_tau = 2.0
true_rho = 0.9
beta = 1.0
noise_sd = 2.0

results_precise = []

for sim in range(n_sim):
    # Generate connected graph
    G = nx.erdos_renyi_graph(n=n, p=0.05, seed=sim)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    A = nx.to_numpy_array(G)
    degrees = A.sum(axis=1)
    degrees[degrees == 0] = 1
    W_matrix = A / degrees[:, None]
    n_actual = A.shape[0]

    # Simulate data
    X = np.random.normal(0, 1, n_actual)
    T = 0.5 * X + np.random.normal(0, 1, n_actual)
    eps = np.random.normal(0, noise_sd, n_actual)

    # Generate Y using SAR structure
    B = true_tau * T + beta * X + eps
    I = np.eye(n_actual)
    Y = np.linalg.solve(I - true_rho * W_matrix, B)

    # Naive OLS
    model_naive = sm.OLS(Y, sm.add_constant(np.column_stack([T, X]))).fit()
    tau_naive = model_naive.params[1]

    # Correct SAR (known structure)
    Y_star = (I - true_rho * W_matrix) @ Y
    model_correct = sm.OLS(Y_star, sm.add_constant(np.column_stack([T, X]))).fit()
    tau_correct = model_correct.params[1]

    # Store results
    results_precise.append({
        "Simulation": sim,
        "Estimator": "Naive (violates SUTVA)",
        "Tau_hat": tau_naive,
        "Bias": tau_naive - true_tau
    })
    results_precise.append({
        "Simulation": sim,
        "Estimator": "Correct (known SAR)",
        "Tau_hat": tau_correct,
        "Bias": tau_correct - true_tau
    })

# Convert to DataFrame
df_precise = pd.DataFrame(results_precise)

# Plot 1: KDE
plt.figure(figsize=(6, 5))
sns.kdeplot(data=df_precise, x="Tau_hat", hue="Estimator", common_norm=False, alpha=0.7)
plt.axvline(true_tau, color="black", linestyle="--", label="True τ = 2.0")
plt.title("Sampling Distribution with ρ = 0.9")
plt.xlabel("Estimated Treatment Effect (τ̂)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Bias strip plot
plt.figure(figsize=(6, 5))
sns.stripplot(data=df_precise, x="Estimator", y="Bias", jitter=True, alpha=0.6)
sns.pointplot(data=df_precise, x="Estimator", y="Bias", join=False, color='red', markers='D', ci='sd', errwidth=1.5)
plt.axhline(0, color="black", linestyle="--")
plt.title("Bias in Estimated Treatment Effect (ρ = 0.95)")
plt.ylabel("Bias (τ̂ - τ)")
plt.tight_layout()
plt.show()
