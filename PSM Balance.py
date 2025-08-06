import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")
np.random.seed(0)
n = 500

# GOOD CASE: no hidden variables
outcome_control_good = np.random.normal(30, 10, n)
outcome_treat_good_after = np.random.normal(30, 10, n)
ps_control_good = np.random.normal(-0.5, 1, n)
ps_treat_good_before = np.random.normal(1.0, 1, n)

# BAD CASE: hidden confounder not adjusted for
outcome_control_bad = np.random.normal(30, 10, n)
outcome_treat_bad = np.random.normal(50, 10, n)
ps_control_bad = np.random.normal(-1.0, 1, n)
ps_treat_bad = np.random.normal(1.5, 1, n)

# Helper function
def make_df(x, group, varname):
    return pd.DataFrame({varname: x, "group": group})

# Assemble dataframes
df_ps_no_hidden = pd.concat([
    make_df(ps_control_good, "Control", "score"),
    make_df(ps_treat_good_before, "Treatment", "score")
]).reset_index(drop=True)

df_outcome_no_hidden = pd.concat([
    make_df(outcome_control_good, "Control", "outcome"),
    make_df(outcome_treat_good_after, "Treatment", "outcome")
]).reset_index(drop=True)

df_ps_hidden = pd.concat([
    make_df(ps_control_bad, "Control", "score"),
    make_df(ps_treat_bad, "Treatment", "score")
]).reset_index(drop=True)

df_outcome_hidden = pd.concat([
    make_df(outcome_control_bad, "Control", "outcome"),
    make_df(outcome_treat_bad, "Treatment", "outcome")
]).reset_index(drop=True)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Row 1: no hidden variables
sns.kdeplot(data=df_ps_no_hidden[df_ps_no_hidden['group'] == "Control"], x="score", ax=axs[0, 0],
            color="#1f77b4", label="Control")
sns.kdeplot(data=df_ps_no_hidden[df_ps_no_hidden['group'] == "Treatment"], x="score", ax=axs[0, 0],
            color="#ff7f0e", label="Treatment")
axs[0, 0].set_title("Before matching (no hidden variables)")
axs[0, 0].set_xlabel("Covariate")
axs[0, 0].legend(title="Group", loc="upper right")

sns.kdeplot(data=df_outcome_no_hidden[df_outcome_no_hidden['group'] == "Control"], x="outcome", ax=axs[0, 1],
            color="#1f77b4", label="Control")
sns.kdeplot(data=df_outcome_no_hidden[df_outcome_no_hidden['group'] == "Treatment"], x="outcome", ax=axs[0, 1],
            color="#ff7f0e", label="Treatment")
axs[0, 1].set_title("After matching (no hidden variables)")
axs[0, 1].set_xlabel("Covariate")
axs[0, 1].legend(title="Group", loc="upper right")

# Row 2: hidden variable case
sns.kdeplot(data=df_ps_hidden[df_ps_hidden['group'] == "Control"], x="score", ax=axs[1, 0],
            color="#2ca02c", label="Control")
sns.kdeplot(data=df_ps_hidden[df_ps_hidden['group'] == "Treatment"], x="score", ax=axs[1, 0],
            color="#d62728", label="Treatment")
axs[1, 0].set_title("Before matching (hidden variable)")
axs[1, 0].set_xlabel("Covariate")
axs[1, 0].legend(title="Group", loc="upper right")

sns.kdeplot(data=df_outcome_hidden[df_outcome_hidden['group'] == "Control"], x="outcome", ax=axs[1, 1],
            color="#2ca02c", label="Control")
sns.kdeplot(data=df_outcome_hidden[df_outcome_hidden['group'] == "Treatment"], x="outcome", ax=axs[1, 1],
            color="#d62728", label="Treatment")
axs[1, 1].set_title("After matching (hidden variable)")
axs[1, 1].set_xlabel("Covariate")
axs[1, 1].legend(title="Group", loc="upper right")

# Finishing touches
for ax in axs.flatten():
    ax.set_ylabel("Density")

#fig.suptitle("Illustration of Propensity Score Matching With and Without Hidden Confounders", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.5, w_pad=2.5)
plt.show()
