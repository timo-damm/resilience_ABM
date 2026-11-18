# PREPARATIONS
# %%
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── locate & import the ABM ──────────────────────────────────────────────────
ABM_PATH = Path(__file__).parent / "resilience_ABM.py"

spec = importlib.util.spec_from_file_location("resilience_ABM", ABM_PATH)
abm  = importlib.util.module_from_spec(spec)
sys.modules["resilience_ABM"] = abm
spec.loader.exec_module(abm)

adj_matrix  = abm.adj_matrix
nodes_df    = abm.nodes_df
ModelConfig = abm.ModelConfig

# ── forward-compatible log_state wrapper ─────────────────────────────────────
# log_state() gained extra required args (e.g. support_given, support_received).
# This wrapper inspects the live signature and passes 0 for any extras so the
# sensitivity script never needs updating when the ABM signature changes.
import inspect as _inspect
_log_sig    = _inspect.signature(abm.log_state)
_log_extras = [
    p for p in list(_log_sig.parameters)[3:]  # skip G, t, history
    if _log_sig.parameters[p].default is _inspect.Parameter.empty
]
def _call_log_state(G, t, history):
    abm.log_state(G, t, history, *([{}] * len(_log_extras)))

# ── defaults & sweep config ───────────────────────────────────────────────────
FP_VAL      = +0.99
DEFAULT_RES = 0.2   # default initial group resilience
DEFAULT_COB = 0.2   # default initial causes of burnout

NUM_RUNS = 50
T        = 200

sweep_res_group = np.round(np.linspace(-1.0, 1.0, 6), 5)
sweep_cob       = np.round(np.linspace( 0.0, 1.0, 6), 5)
sweep_rep_low   = np.round(np.linspace( 0.0, 0.5, 6), 5)
sweep_rep_high  = np.round(np.linspace( 0.5, 1.0, 6), 5)

# ── core runner – initial-state sweep ────────────────────────────────────────
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
            _call_log_state(G, t, history)

        for k in keys:
            results[k].append(history[k])

    return {k: np.array(v) for k, v in results.items()}


# ── core runner – repression parameter sweep ─────────────────────────────────
def run_repression(adj_matrix, nodes_df, cfg,
                   num_runs: int = NUM_RUNS,
                   T: int = T) -> dict:
    """Run with individual resilience clamped at +1 and default initial state,
    using whatever rep_low / rep_high is already set on cfg."""
    keys = ["group_resilience", "mean_individual_resilience",
            "internal_social_support", "causes_of_burnout",
            "mean_external_social_support", "repression"]
    results = {k: [] for k in keys}

    for _ in range(num_runs):
        G = abm.build_graph(adj_matrix, nodes_df, cfg)
        for n in G.nodes():
            G.nodes[n]["individual_resilience"] = FP_VAL
        # group_resilience and causes_of_burnout stay at build_graph defaults

        history = abm.empty_history()
        for t in range(T):
            G.graph["repression"] = abm.repression_schedule(t, cfg)
            abm.timestep_update(G, cfg)
            _call_log_state(G, t, history)

        for k in keys:
            results[k].append(history[k])

    return {k: np.array(v) for k, v in results.items()}


def mean_std(results: dict, key: str = "mean_individual_resilience"):
    arr = results[key]
    return arr.mean(axis=0), arr.std(axis=0)

# ── run OFAT – initial state ──────────────────────────────────────────────────
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

# ── run OFAT – repression schedule ───────────────────────────────────────────
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

# ── summary CSV ───────────────────────────────────────────────────────────────
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
df.to_csv(Path(__file__).parent / "output" / "ofat_summary_new.csv", index=False)

# %%
# ── plot config ───────────────────────────────────────────────────────────────
PARAM_LABELS = {
    "group_resilience":  "Initial SMO Resilience",
    "causes_of_burnout": "Initial Causes of Burnout",
    "rep_low":           "Repression Low (rep_low)",
    "rep_high":          "Repression High (rep_high)",
}
COLOR = "steelblue"

# ── Figure 1 (per parameter): time-series grid – initial state ───────────────
for param, val_dict in results_store.items():
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
        t = np.arange(T)

        ax.fill_between(t, m_t - s_t, m_t + s_t, alpha=0.2, color=COLOR)
        ax.plot(t, m_t, color=COLOR, lw=1.5)
        ax.axhline(FP_VAL, color="grey",  lw=0.8, ls="--", label="FP (+1)")
        ax.axhline(0,      color="black", lw=0.4, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"init = {val:+.2f}", fontsize=9)
        ax.set_xlabel("Time", fontsize=8)

        if col_j == 0:
            ax.set_ylabel("Mean individual resilience", fontsize=8)

    fig.suptitle(
        f"OFAT – {PARAM_LABELS[param]} (FP = +1)\n"
        f"(mean ± 1 SD, {NUM_RUNS} runs)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = Path(__file__).parent / "output" / f"ofat_{param}_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Figure 2 (per repression param): time-series grid ────────────────────────
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
    out = Path(__file__).parent / "output" /  f"ofat_{rep_param}_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Figure 3: displacement summary – all four parameters ─────────────────────
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
out = Path(__file__).parent / "output" / "ofat_displacement_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
# =============================================================================
# GLOBAL SENSITIVITY ANALYSIS - OLS with effect sizes that do not depend on N
# =============================================================================
# Inputs (Latin hypercube sample, 10 000 runs):
#   init_indiv_res ~ U(-1, 1)     init_group_res ~ U(-1, 1)
#   init_cob       ~ U( 0, 1)
#   rep_low        ~ U( 0, 0.5)   rep_high       ~ U(0.5, 1)
# Outcome: mean individual resilience at t = 100 (single stochastic run).
#
# Reported per predictor:
#   coef        raw OLS coefficient (change in outcome per 1 SD of the input)
#   beta        standardised coefficient (change in outcome-SD per 1 SD input)
#   partial R2  share of the variance left over by the other predictors
#   f2          Cohen's f2 = partial R2 / (1 - partial R2)
#   delta R2    unique variance (R2_full - R2_without_predictor)
#   S1          non-parametric first-order index Var(E[Y|X_i]) / Var(Y),
#               captures non-linear main effects
# All with bootstrap 95% CIs (resampling runs).
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import qmc

GSA_N  = 10_000
GSA_T  = 100
N_BOOT = 2_000
SEED   = 42

FEATURE_COLS = ["init_indiv_res", "init_group_res", "init_cob",
                "rep_low", "rep_high"]
LOWER = [-1.0, -1.0, 0.0, 0.0, 0.5]
UPPER = [ 1.0,  1.0, 1.0, 0.5, 1.0]

FEATURE_LABELS = {
    "init_indiv_res": "Initial\nIndiv. Resilience",
    "init_group_res": "Initial\nGroup Resilience",
    "init_cob":       "Initial\nCauses of Burnout",
    "rep_low":        "Repression\nLow",
    "rep_high":       "Repression\nHigh",
}
LATEX_LABELS = {
    "init_indiv_res": r"Individual Resilience ($\hat{r}$)",
    "init_group_res": r"Group Resilience ($R$)",
    "init_cob":       r"Causes of Burnout ($B$)",
    "rep_low":        r"Stressors Minimum ($\rho_{min}$)",
    "rep_high":       r"Stressors Maximum ($\rho_{max}$)",
}

# ── Latin hypercube sample (replaces independent uniform draws) ───────────────
sampler = qmc.LatinHypercube(d=len(FEATURE_COLS), seed=SEED)
sample  = qmc.scale(sampler.random(GSA_N), LOWER, UPPER)   # shape (GSA_N, 5)

gsa_init_indiv, gsa_init_group, gsa_init_cob, gsa_rep_low, gsa_rep_high = sample.T

gsa_outcomes = np.empty(GSA_N)

print(f"\nRunning {GSA_N} global sensitivity runs ...")
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

    for t in range(GSA_T):
        G.graph["repression"] = abm.repression_schedule(t, cfg_gsa)
        abm.timestep_update(G, cfg_gsa)

    nodes_left = list(G.nodes())
    if nodes_left:
        gsa_outcomes[i] = np.mean(
            [G.nodes[n]["individual_resilience"] for n in nodes_left]
        )
    else:
        gsa_outcomes[i] = np.nan   # group dissolved

# ── data frames: all runs (for survival) and valid runs (for OLS) ────────────
gsa_all = pd.DataFrame(sample, columns=FEATURE_COLS)
gsa_all["outcome"]  = gsa_outcomes
gsa_all["survived"] = (~np.isnan(gsa_outcomes)).astype(int)
gsa_df = gsa_all.dropna().reset_index(drop=True)

n_obs = len(gsa_df)
k     = len(FEATURE_COLS)
print(f"\nValid runs: {n_obs} / {GSA_N}  "
      f"(dissolved: {GSA_N - n_obs}, {100 * (GSA_N - n_obs) / GSA_N:.1f}%)")

X_raw = gsa_df[FEATURE_COLS].values
y     = gsa_df["outcome"].values


def standardise(a):
    return (a - a.mean(axis=0)) / a.std(axis=0)


X_std = standardise(X_raw)

# ── 1. Classic OLS (raw outcome, standardised inputs) ────────────────────────
ols_model = sm.OLS(y, sm.add_constant(X_std)).fit()
print("\n-- OLS Results (standardised inputs, raw outcome) --")
print(ols_model.summary())
ci_raw = np.asarray(ols_model.conf_int())[1:]          # drop intercept
coef   = np.asarray(ols_model.params)[1:]
R2_full = ols_model.rsquared


# ── 2. Effect-size helpers ───────────────────────────────────────────────────
def ols_effects(X, y):
    """Standardised beta and partial R2 from OLS on re-standardised X and y.
    Returns beta (k,), partial_r2 (k,), R2 (float)."""
    n, kk = X.shape
    Xz = standardise(X)
    yz = (y - y.mean()) / y.std()
    Z  = np.column_stack([np.ones(n), Xz])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    b     = ZtZ_inv @ Z.T @ yz
    resid = yz - Z @ b
    df    = n - kk - 1
    s2    = resid @ resid / df
    se    = np.sqrt(s2 * np.diag(ZtZ_inv))
    t     = b / se
    pr2   = t[1:] ** 2 / (t[1:] ** 2 + df)
    R2    = 1.0 - (resid @ resid) / n          # sum(yz^2) = n
    return b[1:], pr2, R2


def first_order_index(x, y, bins=20):
    """Bias-corrected Var(E[Y|X])/Var(Y) from equal-count bins."""
    order  = np.argsort(x)
    groups = np.array_split(order, bins)
    ybar   = y.mean()
    ssb    = sum(len(g) * (y[g].mean() - ybar) ** 2 for g in groups)
    sst    = ((y - ybar) ** 2).sum()
    msw    = (sst - ssb) / (len(y) - bins)
    return (ssb - (bins - 1) * msw) / sst


# ── 3. Point estimates ───────────────────────────────────────────────────────
beta, pr2, R2 = ols_effects(X_raw, y)
f2     = pr2 / (1.0 - pr2)
dR2    = f2 * (1.0 - R2)                 # unique variance explained
S1     = np.array([first_order_index(X_raw[:, j], y) for j in range(k)])

# ── 4. Bootstrap CIs (resample runs) ─────────────────────────────────────────
rng_b = np.random.default_rng(SEED)
boot_beta = np.empty((N_BOOT, k))
boot_pr2  = np.empty((N_BOOT, k))
boot_S1   = np.empty((N_BOOT, k))
for bi in range(N_BOOT):
    idx = rng_b.integers(0, n_obs, n_obs)
    Xb, yb = X_raw[idx], y[idx]
    boot_beta[bi], boot_pr2[bi], _ = ols_effects(Xb, yb)
    boot_S1[bi] = [first_order_index(Xb[:, j], yb) for j in range(k)]

q = lambda a: np.percentile(a, [2.5, 97.5], axis=0)
beta_lo, beta_hi = q(boot_beta)
pr2_lo,  pr2_hi  = q(boot_pr2)
f2_lo, f2_hi     = pr2_lo / (1 - pr2_lo), pr2_hi / (1 - pr2_hi)
S1_lo,  S1_hi    = q(boot_S1)

effects = pd.DataFrame({
    "parameter":  FEATURE_COLS,
    "coef":       coef,
    "coef_lo":    ci_raw[:, 0],
    "coef_hi":    ci_raw[:, 1],
    "beta":       beta,    "beta_lo": beta_lo, "beta_hi": beta_hi,
    "partial_r2": pr2,     "pr2_lo":  pr2_lo,  "pr2_hi":  pr2_hi,
    "f2":         f2,      "f2_lo":   f2_lo,   "f2_hi":   f2_hi,
    "delta_r2":   dR2,
    "S1":         S1,      "S1_lo":   S1_lo,   "S1_hi":   S1_hi,
})

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(f"\n-- Effect sizes (R2 = {R2_full:.4f}, N = {n_obs}) --")
print(effects.round(4).to_string(index=False))
print(f"\nSD of outcome (for reference): {y.std(ddof=1):.4f}")
print(f"Sum of unique dR2 = {dR2.sum():.4f}  (vs full R2 = {R2_full:.4f})")
print(f"Sum of S1         = {S1.sum():.4f}")
effects.to_csv(Path(__file__).parent / "output" / "gsa_effect_sizes.csv", index=False)

# LaTeX-ready table rows
print("\n-- LaTeX rows: coef [CI] & beta [CI] & partial R2 & f2 [CI] & S1 --")
for _, r in effects.iterrows():
    print(
        f"{LATEX_LABELS[r.parameter]} & {r.coef:.3f} [{r.coef_lo:.3f}, {r.coef_hi:.3f}] "
        f"& {r.beta:.2f} [{r.beta_lo:.2f}, {r.beta_hi:.2f}] "
        f"& {r.partial_r2:.2f} & {r.f2:.2f} [{r.f2_lo:.2f}, {r.f2_hi:.2f}] "
        f"& {r.S1:.2f} \\\\"
    )

# ── 5. Non-linearity check: add squared terms ────────────────────────────────
X_quad  = np.column_stack([X_std, X_std ** 2])
quad    = sm.OLS(y, sm.add_constant(X_quad)).fit()
print("\n-- Non-linearity check --")
print(f"R2 linear = {R2_full:.4f} | R2 with squared terms = {quad.rsquared:.4f} "
      f"| gain = {quad.rsquared - R2_full:.4f}")
for name, c, p in zip(FEATURE_COLS, np.asarray(quad.params)[k + 1:],
                      np.asarray(quad.pvalues)[k + 1:]):
    print(f"  squared {name:15s} coef = {c:+.4f}  (p = {p:.2g})")

# ── 6. Survival model: which parameters drive group dissolution? ─────────────
X_all_std = standardise(gsa_all[FEATURE_COLS].values)
logit = sm.Logit(gsa_all["survived"].values, sm.add_constant(X_all_std)).fit(disp=0)
print("\n-- Logit for survival (1 = group survived), standardised inputs --")
print(f"McFadden pseudo-R2 = {logit.prsquared:.4f}")
for name, b_, lo, hi in zip(FEATURE_COLS, np.asarray(logit.params)[1:],
                            *np.asarray(logit.conf_int())[1:].T):
    print(f"  {name:15s} OR per SD = {np.exp(b_):.2f}  "
          f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}]")

# ── Figure 4: standardised coefficients with bootstrap CIs ───────────────────
coef_df = effects.copy()
coef_df["label"] = coef_df["parameter"].map(FEATURE_LABELS)
bar_colors = ["tomato" if b_ < 0 else "steelblue" for b_ in coef_df["beta"]]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(coef_df["label"], coef_df["beta"],
              color=bar_colors, edgecolor="white", linewidth=0.6)
ax.errorbar(coef_df["label"], coef_df["beta"],
            yerr=[coef_df["beta"] - coef_df["beta_lo"],
                  coef_df["beta_hi"] - coef_df["beta"]],
            fmt="none", color="black", capsize=4, lw=1.2)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel(r"Standardised coefficient ($\beta$)", fontsize=10)
ax.set_title(
    f"Global Sensitivity Analysis - Effect on Mean Individual Resilience at T={GSA_T}\n"
    f"OLS on {n_obs:,} runs  |  R² = {R2_full:.3f}  |  bars: 95% bootstrap CI",
    fontsize=11,
)
for bar, (_, r) in zip(bars, coef_df.iterrows()):
    if r.beta >= 0:
        ypos, va = r.beta_hi + 0.015, "bottom"
    else:
        ypos, va = r.beta_lo - 0.015, "top"
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{r.beta:.2f}", ha="center", va=va, fontsize=9)
ax.margins(y=0.15)

plt.tight_layout()
out = Path(__file__).parent / "output" / "gsa_ols_coefficients.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out.name}")

# ── Figure 5: partial regression plots (with LOWESS to show non-linearity) ───
fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=True)

for ax, (col_i, feat) in zip(axes, enumerate(FEATURE_COLS)):
    others   = [j for j in range(k) if j != col_i]
    X_others = sm.add_constant(X_std[:, others])

    res_y    = sm.OLS(y,               X_others).fit().resid
    res_feat = sm.OLS(X_std[:, col_i], X_others).fit().resid

    order   = np.argsort(res_feat)
    bin_idx = np.array_split(order, 40)
    bx = [res_feat[b].mean() for b in bin_idx]
    by = [res_y[b].mean()    for b in bin_idx]
    ax.scatter(bx, by, s=18, alpha=0.7, color="steelblue", edgecolors="none")

    slope = np.polyfit(res_feat, res_y, 1)
    xline = np.linspace(res_feat.min(), res_feat.max(), 100)
    ax.plot(xline, np.polyval(slope, xline), color="tomato", lw=1.5, label="linear")

    lw_fit = sm.nonparametric.lowess(res_y, res_feat, frac=0.3)
    ax.plot(lw_fit[:, 0], lw_fit[:, 1], color="black", lw=1.2, ls="--", label="LOWESS")

    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
    if col_i == 0:
        ax.set_ylabel("Residual outcome", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{FEATURE_LABELS[feat]}\npartial R² = {pr2[col_i]:.3f}", fontsize=9)

fig.suptitle(
    "Partial Regression Plots - Global Sensitivity Analysis\n"
    "(binned means, each feature partialled out against the rest)",
    fontsize=11,
)
plt.tight_layout()
out = Path(__file__).parent / "output" / "output" /  "gsa_partial_regression.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out.name}")
# %%
