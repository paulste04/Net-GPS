import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 1  SET-UP
# ----------------------------------------
sns.set(style="whitegrid")
np.random.seed(42)

n, n_show = 100, 12                # total individuals & those we draw
t_grid    = np.linspace(0, 10, 200)

# Individual nonlinear (√) dose–response parameters
intercepts = np.random.normal(2.0, 0.5, n)
slopes     = np.random.normal(1.2, 0.2, n)
f          = lambda a, b, t: a + b*np.sqrt(t)

# Potential-outcome curves and population mean μ(t)
Y_curves = np.array([f(a, b, t_grid) for a, b in zip(intercepts, slopes)])
mu_t     = Y_curves.mean(axis=0)

# ----------------------------------------
# 2  BINARY STEP POTENTIAL OUTCOMES
# ----------------------------------------
x_ctrl, x_mid, x_treat = 0.25, 0.50, 0.75   # x-positions for the step plot
Y0 = intercepts                              # outcomes if T = 0
Y1 = intercepts + slopes*np.sqrt(10)         # outcomes if T = 1
y0_mean, y1_mean = Y0.mean(), Y1.mean()

# ----------------------------------------
# 3  PLOTTING
# ----------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# ---- LEFT PLOT (continuous, nonlinear μ(t)) --------------------
for i in range(n_show):
    axs[0].plot(t_grid, Y_curves[i],
                color="lightsteelblue", alpha=0.6, lw=1,
                label="Individual $Y_i(t)$" if i == 0 else "")

axs[0].plot(t_grid, mu_t,
            color="royalblue", lw=3, label=r"Population $\mu(t)$")

axs[0].set_xlabel(r"Treatment level $t$")
axs[0].set_ylabel(r"Outcome $Y$")
axs[0].set_title("Continuous Treatment")
axs[0].legend()

# ---- RIGHT PLOT (binary, step functions) ----------------------
# thin grey step functions for all individuals
for y0, y1 in zip(Y0, Y1):
    axs[1].hlines(y0, x_ctrl, x_mid, color="lightgray", alpha=0.3, lw=1)
    axs[1].vlines(x_mid, y0, y1,   color="lightgray", alpha=0.3, lw=1)
    axs[1].hlines(y1, x_mid, x_treat, color="lightgray", alpha=0.3, lw=1)

# highlight one representative individual
axs[1].hlines(Y0[0], x_ctrl, x_mid, color="lightgray", lw=1)
axs[1].vlines(x_mid, Y0[0], Y1[0], color="lightgray", lw=1)
axs[1].hlines(Y1[0], x_mid, x_treat, color="lightgray", lw=1,
              label="Individual potential\noutcome")

# thick population step (ATE)
axs[1].hlines(y0_mean, x_ctrl, x_mid, color="darkgreen", lw=4)
axs[1].vlines(x_mid,  y0_mean, y1_mean, color="darkgreen", lw=4)
axs[1].hlines(y1_mean, x_mid, x_treat, color="darkgreen", lw=4,
              label="Average treatment effect")

# axis cosmetics
axs[1].set_xticks([x_ctrl, x_treat])
axs[1].set_xticklabels(["Control ($T=0$)", "Treated ($T=1$)"])
axs[1].set_xlim(0, 1)
axs[1].set_xlabel("Treatment group")
axs[1].set_ylabel(r"Outcome $Y$")
axs[1].set_title("Binary Treatment")
axs[1].legend()

plt.tight_layout()
plt.show()
