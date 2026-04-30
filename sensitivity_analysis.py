# PREPARATIONS
# %%
import importlib.util
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# importing ABM to run it
ABM_PATH = Path(__file__).parent / "resilience_ABM.py"

spec = importlib.util.spec_from_file_location("resilience_ABM", ABM_PATH)
abm  = importlib.util.module_from_spec(spec)
sys.modules["resilience_ABM"] = abm
spec.loader.exec_module(abm)

adj_matrix  = abm.adj_matrix
nodes_df    = abm.nodes_df
ModelConfig = abm.ModelConfig

# config for sweeping
FP_VAL      = +1.0
DEFAULT_RES = 0.2  
DEFAULT_COB = 0.2   

NUM_RUNS = 100
T        = 100

sweep_res_group = np.round(np.arange(-1.0, 1.0 + 0.1, 0.1), 5)
sweep_cob       = np.round(np.arange( 0.0, 1.0 + 0.1, 0.1), 5)
sweep_rep_low   = np.round(np.arange( 0.0, 0.5 + 0.1, 0.1), 5)
sweep_rep_high  = np.round(np.arange( 0.5, 1.0 + 0.1, 0.1), 5)

# OFAT SENSITIVITY ANALYSIS
# %%
# resilience and COB OFAT parameter sweeping function
def run_clamped(adj_matrix, nodes_df, cfg,
                init_res_group: float,
                init_cob: float,
                num_runs: int = NUM_RUNS,
                T: int = T) -> dict:
    keys = ["group_resilience", "mean_individual_resilience",
            "internal_social_support", "causes_of_burnout",
            "mean_external_social_support", "repression"]
    results = {k: [] for k in keys}

    for _ in range(num_runs):
        G = abm.build_graph(adj_matrix, nodes_df, cfg)
        for n in G.nodes():
            G.nodes[n]["individual_resilience"] = FP_VAL
        G.graph["group_resilience"]  = init_res_group
        G.graph["causes_of_burnout"] = init_cob

        history = abm.empty_history()
        for t in range(T):
            G.graph["repression"] = abm.repression_schedule(t, cfg)
            abm.timestep_update(G, cfg)
            abm.log_state(G, t, history)

        for k in keys:
            results[k].append(history[k])

    return {k: np.array(v) for k, v in results.items()}


# repression OFAT parameter sweeping function
def run_repression(adj_matrix, nodes_df, cfg,
                   num_runs: int = NUM_RUNS,
                   T: int = T) -> dict:
    keys = ["group_resilience", "mean_individual_resilience",
            "internal_social_support", "causes_of_burnout",
            "mean_external_social_support", "repression"]
    results = {k: [] for k in keys}

    for _ in range(num_runs):
        G = abm.build_graph(adj_matrix, nodes_df, cfg)
        for n in G.nodes():
            G.nodes[n]["individual_resilience"] = FP_VAL

        history = abm.empty_history()
        for t in range(T):
            G.graph["repression"] = abm.repression_schedule(t, cfg)
            abm.timestep_update(G, cfg)
            abm.log_state(G, t, history)

        for k in keys:
            results[k].append(history[k])

    return {k: np.array(v) for k, v in results.items()}


def mean_std(results: dict, key: str = "mean_individual_resilience"):
    arr = results[key]
    return arr.mean(axis=0), arr.std(axis=0)

# OFAT for group resilience and causes of burnout
cfg = ModelConfig()
results_store: dict = {}

state_params = {
    "group_resilience":  sweep_res_group,
    "causes_of_burnout": sweep_cob,
}

total = sum(len(v) for v in state_params.values())
done  = 0

for param, values in state_params.items():
    results_store[param] = {}
    for val in values:
        done += 1
        print(f"[initial-state] {param} = {val:+.3f}  ({done}/{total})")

        init_res_group = val if param == "group_resilience"  else DEFAULT_RES
        init_cob       = val if param == "causes_of_burnout" else DEFAULT_COB

        results_store[param][float(val)] = run_clamped(
            adj_matrix, nodes_df, cfg,
            init_res_group=init_res_group,
            init_cob=init_cob,
        )

# OFAT for repression (sweeping low values and high values, not changing time)
rep_results: dict = {
    "rep_low":  {},
    "rep_high": {},
}

total_rep = len(sweep_rep_low) + len(sweep_rep_high)
done_rep  = 0

for val in sweep_rep_low:
    done_rep += 1
    print(f"[repression] rep_low = {val:.2f}  ({done_rep}/{total_rep})")
    cfg_rep = ModelConfig()
    cfg_rep.rep_low = float(val)
    rep_results["rep_low"][float(val)] = run_repression(
        adj_matrix, nodes_df, cfg_rep)

for val in sweep_rep_high:
    done_rep += 1
    print(f"[repression] rep_high = {val:.2f}  ({done_rep}/{total_rep})")
    cfg_rep = ModelConfig()
    cfg_rep.rep_high = float(val)
    rep_results["rep_high"][float(val)] = run_repression(
        adj_matrix, nodes_df, cfg_rep)

# saving all outputs to a CSV
rows = []

# initial-state rows
for param, val_dict in results_store.items():
    default_init = DEFAULT_RES if param == "group_resilience" else DEFAULT_COB
    for val, res in val_dict.items():
        m_t, s_t = mean_std(res, "mean_individual_resilience")
        g_t, _   = mean_std(res, "group_resilience")
        rows.append({
            "sweep_type":           "initial_state",
            "parameter":            param,
            "init_value":           val,
            "default_init_value":   default_init,
            "delta":                round(val - default_init, 5),
            "final_mean_indiv_res": round(m_t[-1], 4),
            "final_std_indiv_res":  round(s_t[-1], 4),
            "final_group_res":      round(g_t[-1], 4),
            "displacement":         round(m_t[-1] - FP_VAL, 4),
        })

# repression rows
REP_DEFAULTS = {"rep_low": 0.2, "rep_high": 0.8}
for rep_param, val_dict in rep_results.items():
    default_val = REP_DEFAULTS[rep_param]
    for val, res in val_dict.items():
        m_t, s_t = mean_std(res, "mean_individual_resilience")
        g_t, _   = mean_std(res, "group_resilience")
        rows.append({
            "sweep_type":           "repression_schedule",
            "parameter":            rep_param,
            "init_value":           val,
            "default_init_value":   default_val,
            "delta":                round(val - default_val, 5),
            "final_mean_indiv_res": round(m_t[-1], 4),
            "final_std_indiv_res":  round(s_t[-1], 4),
            "final_group_res":      round(g_t[-1], 4),
            "displacement":         round(m_t[-1] - FP_VAL, 4),
        })

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "ofat_summary.csv", index=False)

# VISUALISATIONS
# %%
#plot config
PARAM_LABELS = {
    "group_resilience":  "Initial Group Resilience",
    "causes_of_burnout": "Initial Causes of Burnout",
    "rep_low":           "Stressors Low",
    "rep_high":          "Stressors High",
}
COLOR = "steelblue"

# resilience and cob OFAT
for param, val_dict in results_store.items():
    vals = sorted(val_dict.keys())
    n_vals = len(vals)

    n_cols = min(5, n_vals)
    n_rows = int(np.ceil(n_vals / 5)) #only to make it readable in article

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 4 * n_rows),
        sharey=True, sharex=True,
    )

    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, val in enumerate(vals):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        m_t, s_t = mean_std(val_dict[val], "mean_individual_resilience")
        t = np.arange(T)

        ax.fill_between(t, m_t - s_t, m_t + s_t, alpha=0.2, color=COLOR)
        ax.plot(t, m_t, color=COLOR, lw=1.5)
        ax.axhline(FP_VAL, color="grey", lw=0.8, ls="--", label="FP (+1)")
        ax.axhline(0, color="black", lw=0.4, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"init = {val:+.2f}", fontsize=9)
        ax.set_xlabel("Time", fontsize=8)

        if col == 0:
            ax.set_ylabel("Mean individual resilience", fontsize=8)

    for idx in range(n_vals, n_rows * n_cols):
        fig.delaxes(axes.flatten()[idx])

    fig.suptitle(
        f"OFAT – {PARAM_LABELS[param]} (FP = +1)\n"
        f"(mean ± 1 SD, {NUM_RUNS} runs)",
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    out = Path(__file__).parent / f"ofat_{param}_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# repression OFAT
for rep_param, val_dict in rep_results.items():
    vals   = sorted(val_dict.keys())
    n_vals = len(vals)

    fig, axes = plt.subplots(
        1, n_vals,
        figsize=(3 * n_vals, 4),
        sharey=True, sharex=True,
    )
    axes = np.atleast_1d(axes)

    for col_j, val in enumerate(vals):
        ax = axes[col_j]
        m_t, s_t = mean_std(val_dict[val], "mean_individual_resilience")
        rep_t, _ = mean_std(val_dict[val], "repression")
        t = np.arange(T)

        ax.fill_between(t, m_t - s_t, m_t + s_t, alpha=0.2, color=COLOR)
        ax.plot(t, m_t,   color=COLOR,     lw=1.5, label="mean indiv. res.")
        ax.plot(t, rep_t, color="dimgrey", lw=1.0, ls="--", label="repression")
        ax.axhline(FP_VAL, color="grey",  lw=0.8, ls=":", label="FP (+1)")
        ax.axhline(0,      color="black", lw=0.4, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{rep_param} = {val:.2f}", fontsize=9)
        ax.set_xlabel("Time", fontsize=8)

        if col_j == 0:
            ax.set_ylabel("Mean individual resilience", fontsize=8)
        if col_j == n_vals - 1:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"OFAT – {PARAM_LABELS[rep_param]} (FP = +1)\n"
        f"(mean ± 1 SD, {NUM_RUNS} runs)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = Path(__file__).parent / f"ofat_{rep_param}_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# displacement summary for all parameters
all_params = list(results_store.items()) + [
    (rp, val_dict) for rp, val_dict in rep_results.items()
]
REP_DEFAULTS = {"rep_low": 0.2, "rep_high": 0.8}

fig, axes = plt.subplots(1, len(all_params),
                         figsize=(5 * len(all_params), 4))
axes = np.atleast_1d(axes)

for ax, (param, val_dict) in zip(axes, all_params):
    if param in ("rep_low", "rep_high"):
        default_init = REP_DEFAULTS[param]
    elif param == "group_resilience":
        default_init = DEFAULT_RES
    else:
        default_init = DEFAULT_COB

    vals = sorted(val_dict.keys())
    displacements = [
        mean_std(val_dict[v], "mean_individual_resilience")[0][-1] - FP_VAL
        for v in vals
    ]
    ax.plot(vals, displacements, marker="o", color=COLOR, lw=1.8, ms=5)
    ax.axhline(0, color="grey",  lw=0.7, ls="--")
    ax.axvline(default_init, color="black", lw=0.7, ls=":", label="default")
    ax.set_xlabel(PARAM_LABELS[param], fontsize=9)
    ax.set_ylabel("Displacement from FP at T=100", fontsize=9)
    ax.set_title(PARAM_LABELS[param], fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle(
    f"OFAT – Final Displacement from Positive Fixed Point\n"
    f"(mean, {NUM_RUNS} runs, T={T})",
    fontsize=11,
)
plt.tight_layout()
out = Path(__file__).parent / "ofat_displacement_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)

# GLOBAL SENSITIVITY ANALYSIS (OLS) 
# %% 
GSA_N = 10000 # number of runs in the sample 
rng   = np.random.default_rng(seed=42)
 
# Latin Hypercube sampling (better for near-random draws from multi-dimensional distribution)
gsa_init_indiv = rng.uniform(-1.0,  1.0, GSA_N)
gsa_init_group = rng.uniform(-1.0,  1.0, GSA_N)
gsa_init_cob   = rng.uniform( 0.0,  1.0, GSA_N)
gsa_rep_low    = rng.uniform( 0.0,  0.5, GSA_N)
gsa_rep_high   = rng.uniform( 0.5,  1.0, GSA_N)
 
gsa_init_indiv = rng.uniform(-1.0,  1.0, GSA_N)
gsa_init_group = rng.uniform(-1.0,  1.0, GSA_N)
gsa_init_cob   = rng.uniform( 0.0,  1.0, GSA_N)
gsa_rep_low    = rng.uniform( 0.0,  0.5, GSA_N)
gsa_rep_high   = rng.uniform( 0.5,  1.0, GSA_N)
 
gsa_outcomes = np.empty(GSA_N)
  
for i in range(GSA_N):
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{GSA_N}")
 
    cfg_gsa          = ModelConfig()
    cfg_gsa.rep_low  = float(gsa_rep_low[i])
    cfg_gsa.rep_high = float(gsa_rep_high[i])
 
    G = abm.build_graph(adj_matrix, nodes_df, cfg_gsa)
    for n in G.nodes():
        G.nodes[n]["individual_resilience"] = float(gsa_init_indiv[i])
    G.graph["group_resilience"]  = float(gsa_init_group[i])
    G.graph["causes_of_burnout"] = float(gsa_init_cob[i])
 
    for t in range(T):
        G.graph["repression"] = abm.repression_schedule(t, cfg_gsa)
        abm.timestep_update(G, cfg_gsa)
 
    nodes_left = list(G.nodes())
    if nodes_left:
        gsa_outcomes[i] = np.mean(
            [G.nodes[n]["individual_resilience"] for n in nodes_left]
        )
    else:
        gsa_outcomes[i] = np.nan   # group dissolved
 
# building design matrix and running ols regressiono
gsa_df = pd.DataFrame({
    "init_indiv_res": gsa_init_indiv,
    "init_group_res": gsa_init_group,
    "init_cob":       gsa_init_cob,
    "rep_low":        gsa_rep_low,
    "rep_high":       gsa_rep_high,
    "outcome":        gsa_outcomes,
}).dropna()
 
FEATURE_COLS = ["init_indiv_res", "init_group_res", "init_cob",
                "rep_low", "rep_high"]
 
# standardise inputs (zero mean, unit variance) so coefficients are comparable
X_raw = gsa_df[FEATURE_COLS].values
X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y     = gsa_df["outcome"].values
 
X_with_const = sm.add_constant(X_std)
ols_model    = sm.OLS(y, X_with_const).fit()
 
print(ols_model.summary())
 
ci      = np.asarray(ols_model.conf_int())   # shape (n_params, 2)
ols_results = pd.DataFrame({
    "parameter":   ["intercept"] + FEATURE_COLS,
    "coefficient": np.asarray(ols_model.params).ravel(),
    "std_err":     np.asarray(ols_model.bse).ravel(),
    "t_stat":      np.asarray(ols_model.tvalues).ravel(),
    "p_value":     np.asarray(ols_model.pvalues).ravel(),
    "ci_low":      ci[:, 0],
    "ci_high":     ci[:, 1],
})
print("R² =", round(ols_model.rsquared, 4),
      "Adj-R² =", round(ols_model.rsquared_adj, 4))

# %% 
# standardised regression coefficients figure
coef_df = ols_results[ols_results["parameter"] != "intercept"].copy()
 
FEATURE_LABELS = {
    "init_indiv_res": "Initial\nIndiv. Resilience",
    "init_group_res": "Initial\nGroup Resilience",
    "init_cob":       "Initial\nCauses of Burnout",
    "rep_low":        "Stressors\nLow",
    "rep_high":       "Stressors\nHigh",
}
coef_df["label"] = coef_df["parameter"].map(FEATURE_LABELS)
 
bar_colors = ["tomato" if c < 0 else "steelblue"
              for c in coef_df["coefficient"]]
 
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(
    coef_df["label"], coef_df["coefficient"],
    color=bar_colors, edgecolor="white", linewidth=0.6,
)
ax.errorbar(
    coef_df["label"],
    coef_df["coefficient"],
    yerr=1.96 * coef_df["std_err"],
    fmt="none", color="black", capsize=4, lw=1.2,
)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Standardised OLS coefficient", fontsize=10)
ax.set_title(
    f"Global Sensitivity Analysis – Effect on Mean Individual Resilience at T={T}\n"
    f"OLS on {len(gsa_df):,} runs  |  R² = {ols_model.rsquared:.3f}",
    fontsize=11,
)
 
for bar, (_, row) in zip(bars, coef_df.iterrows()):
    p = row["p_value"]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    y_pos = row["coefficient"] + 1.96 * row["std_err"]
    y_pos += 0.005 if y_pos >= 0 else -0.015
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            sig, ha="center", va="bottom", fontsize=9) # p-values below each bar
 
plt.tight_layout()
out = Path(__file__).parent / "gsa_ols_coefficients.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out.name}")
 
# %% 
#Partial regression plots
partial_r2 = {}

fig, axes = plt.subplots(1, len(FEATURE_COLS),
                         figsize=(4 * len(FEATURE_COLS), 4),
                         sharey=True)
 
for ax, (col_i, feat) in zip(axes, enumerate(FEATURE_COLS)):
    # residualise y and the focal feature against all other features
    others = [j for j in range(len(FEATURE_COLS)) if j != col_i]
    X_others = sm.add_constant(X_std[:, others])
 
    res_y    = sm.OLS(y,              X_others).fit().resid
    res_feat = sm.OLS(X_std[:, col_i], X_others).fit().resid

    pr2 = np.corrcoef(res_feat, res_y)[0, 1] ** 2
    partial_r2[feat] = pr2

 
    # bin into 40 quantile bins for a clean scatter
    order   = np.argsort(res_feat)
    bin_idx = np.array_split(order, 40)
    bx = [res_feat[b].mean() for b in bin_idx]
    by = [res_y[b].mean()    for b in bin_idx]
 
    ax.scatter(bx, by, s=18, alpha=0.7, color="steelblue", edgecolors="none")
 
    # overlay OLS line
    slope = np.polyfit(res_feat, res_y, 1)
    xline = np.linspace(res_feat.min(), res_feat.max(), 100)
    ax.plot(xline, np.polyval(slope, xline), color="tomato", lw=1.5)
 
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
    if col_i == 0:
        ax.set_ylabel("Residual outcome", fontsize=9)
    ax.set_title(FEATURE_LABELS[feat], fontsize=9)
 
fig.suptitle(
    "Partial Regression Plots – Global Sensitivity Analysis\n", #binned features, means each feature is paralleled out against the rest
    fontsize=11,
)
plt.tight_layout()
out = Path(__file__).parent / "gsa_partial_regression.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out.name}")


# %%
#partial R squared
for feat, pr2 in partial_r2.items():
    print(f"  {FEATURE_LABELS[feat].replace(chr(10), ' '):35s}  partial R² = {pr2:.4f}")

# %%
