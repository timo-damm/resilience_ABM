# %%
import importlib.util
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

# --- adjust this import to match your ABM script's filename/module ---
# --- load the ABM module the same way your other script does ---
ABM_PATH = Path(__file__).parent / "resilience_ABM.py"
 
spec = importlib.util.spec_from_file_location("resilience_ABM", ABM_PATH)
abm = importlib.util.module_from_spec(spec)
sys.modules["resilience_ABM"] = abm
spec.loader.exec_module(abm)
 
adj_matrix = abm.adj_matrix
nodes_df = abm.nodes_df
ModelConfig = abm.ModelConfig
build_graph = abm.build_graph
repression_schedule = abm.repression_schedule
timestep_update = abm.timestep_update
log_state = abm.log_state
empty_history = abm.empty_history
run_all = abm.run_all
 
cfg = ModelConfig()
 
 
def sat(x):
    return 1 - x**2


#%%
# ---------------------------------------------------------------------
# ODE system, now forced by the real (time-varying) repression schedule
# instead of a fixed rho
# ---------------------------------------------------------------------
def iss_hat_fn(rhat, rho, cfg, k_mean, ess_hat, alpha=0.015, eps=0.0):
    rhat = np.asarray(rhat, dtype=float)
    need = cfg.tau - ess_hat - rhat
    gate = (need > 0).astype(float) if eps == 0 else 1 / (1 + np.exp(-need / eps))
    capacity = cfg.max_support_capacity * (rhat + 1) / 2
    return gate * np.maximum(0.0, capacity - k_mean * alpha * rho)


def rhs(t, y, cfg, k_mean, ess_hat, eps=0.01):
    rhat, R, B = y
    rho, alpha, c = repression_schedule(t, cfg), 0.015, cfg.support_cost
    iss = float(iss_hat_fn(rhat, rho, cfg, k_mean, ess_hat, alpha, eps))

    drhat = sat(rhat) * alpha * (-rho + ess_hat + iss + R - B) - c * iss
    dR    = sat(R)    * alpha * (-rho + iss + rhat)
    dB    = sat(B)    * alpha * (-rhat - R)
    if B <= 0 and dB < 0:
        dB = 0.0
    return [drhat, dR, dB]
#%%
# ---------------------------------------------------------------------
# real runs
# ---------------------------------------------------------------------
def run_abm_ensemble(adj_matrix, nodes_df, cfg, min_runs=2):
    results = abm.results          

    n_alive = np.sum(~np.isnan(results["group_resilience"]), axis=0)
    last = np.max(np.where(n_alive >= min_runs)[0])   # cut off once too few runs remain
    sl = slice(0, last + 1)

    out = {}
    for key in ("mean_individual_resilience", "group_resilience",
                "causes_of_burnout", "internal_social_support"):
        arr = results[key][:, sl]                     # (num_runs, T_valid), nan-padded
        p025, p25, p75, p975 = np.nanpercentile(arr, [2.5, 25, 75, 97.5], axis=0)
        out[key] = {
            "mean": np.nanmean(arr, axis=0),
            "median": np.nanmedian(arr, axis=0),
            "p025": p025, "p25": p25, "p75": p75, "p975": p975,
        }
    out["n_alive"] = n_alive[sl]
    return out

#%%
# ---------------------------------------------------------------------
# time window: match the ABM's discrete steps exactly
# ---------------------------------------------------------------------
tspan = (0, cfg.T - 1)
t_eval = np.arange(cfg.T)


# ---------------------------------------------------------------------
# background streamlines: grid of ODE trajectories under the real
# repression schedule
# ---------------------------------------------------------------------
n_rhat = 21
n_B = 21
R_fixed = 0.0

rhat_grid_vals = np.linspace(-1, 1, n_rhat)
B_grid_vals = np.linspace(0, 1, n_B)
rhat_mesh, B_mesh = np.meshgrid(rhat_grid_vals, B_grid_vals, indexing="ij")

initial_conditions = np.column_stack([
    rhat_mesh.ravel(),
    np.full(n_rhat * n_B, R_fixed),
    B_mesh.ravel(),
])

fig, ax = plt.subplots(figsize=(9, 7))

cmap = "viridis"

ess_hat = nodes_df["SOC_SUP"].mean()/10
k_mean = 8.0

for y0 in initial_conditions:
    sol = solve_ivp(
        rhs, tspan, y0,
        args=(cfg, k_mean, ess_hat),
        t_eval=t_eval, rtol=1e-8, atol=1e-8,
        max_step=1.0,
    )

    rhat = sol.y[0]
    B = sol.y[2]

    rho_t = np.array([repression_schedule(t, cfg) for t in sol.t])
    iss_hat = iss_hat_fn(rhat, rho_t, cfg, k_mean, ess_hat, eps=0.01)

    points = np.array([B, rhat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    color_norm = plt.Normalize(0, 1)

    lc = LineCollection(segments, cmap=cmap, norm=color_norm, linewidth=2, alpha=0.8)
    lc.set_array(iss_hat[:-1])
    ax.add_collection(lc)

    step = 20      
    span = 3        
    min_move = 0.04

    for i in range(step, len(B) - span, step):
        dB = B[i + span] - B[i]
        dr = rhat[i + span] - rhat[i]
        if np.hypot(dB, dr) < min_move:
            continue
        ax.annotate(
            "",
            xy=(B[i + span], rhat[i + span]),
            xytext=(B[i], rhat[i]),
            arrowprops={
                "arrowstyle": "->",
                "color": "k",
                "lw": 1,
                "mutation_scale": 12,
                "alpha": 0.7,
                "shrinkA": 0,
                "shrinkB": 0,
            },

        )

    ax.plot(B[0], rhat[0], "o", color="k", markersize=2, alpha=0.6)


#ax.set_xlabel("Causes of Burnout", fontsize=14)
#ax.set_ylabel("Resilience", fontsize=14)
#ax.set_title(r"Phase trajectories at $\rho = 0.2$", fontsize=14)

#out = Path(__file__).parent / "phase_2D_replow.png"
#fig.savefig(out, dpi=150, bbox_inches="tight")

ax.add_patch(
    Rectangle(
        (0, -1), 1, 2,
        facecolor="white", alpha=0.6, edgecolor="none", zorder=5,
    )
)

# ---------------------------------------------------------------------
# highlighted trajectory: mean over all ABM runs, with percentile bands
# ---------------------------------------------------------------------
stats = run_abm_ensemble(adj_matrix, nodes_df, cfg)

PINK_95 = "#f8c8dc"   # light pink: 95% band (2.5th-97.5th percentile)
PINK_50 = "#e75480"   # darker pink: 50% band (25th-75th percentile)


def draw_bands(axis, x, s, alpha95=0.6, alpha50=0.6, zorder=8):
    axis.fill_between(x, s["p025"], s["p975"], facecolor=PINK_95,
                      alpha=alpha95, edgecolor="none", zorder=zorder)
    axis.fill_between(x, s["p25"], s["p75"], facecolor=PINK_50,
                      alpha=alpha50, edgecolor="none", zorder=zorder + 0.5)


rhat_s = stats["mean_individual_resilience"]
R_s = stats["group_resilience"]
B_s = stats["causes_of_burnout"]
iss_s = stats["internal_social_support"]

CENTER = "mean"       # or "median"
B_m = B_s[CENTER]
rhat_m = rhat_s[CENTER]
R_m = R_s[CENTER]
iss_m = iss_s[CENTER]

# percentile bands around individual resilience (vertical band at each timestep)
draw_bands(ax, B_m, rhat_s)

# mean trajectory, coloured by mean internal social support
points = np.array([B_m, rhat_m]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=color_norm, linewidth=4, alpha=1.0, zorder=10)
lc.set_array(iss_m[:-1])
ax.add_collection(lc)
ax.plot(B_m[0], rhat_m[0], "ko", markersize=6, zorder=11)
ax.axhline(0.0, linestyle="--", color="grey")
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("Causes of Burnout", fontsize=14)
ax.set_ylabel("Resilience", fontsize=14)
#ax.set_title("Ego-Projection, Reproducing Fieldwork", fontsize=14)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label("Average Internal Social Support", fontsize=14)

# group resilience (dotted) with the same pink bands, fainter
ax_right = ax.twinx()
ax_right.set_ylim(-1, 1)
draw_bands(ax_right, B_m, R_s, alpha95=0.25, alpha50=0.25, zorder=6)
line_right, = ax_right.plot(B_m, R_m, linestyle=":", color="black", linewidth=2,
                            label="Average SMO resilience")
ax_right.set_yticks([])
ax_right.tick_params(right=False, labelright=False)

cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
line_left_proxy = Line2D([0], [0], color=cmap_obj(0.5), linewidth=4,
                         label="Average individual resilience")
band95_proxy = Rectangle((0, 0), 1, 1, facecolor=PINK_95, alpha=0.6, label="95% of runs")
band50_proxy = Rectangle((0, 0), 1, 1, facecolor=PINK_50, alpha=0.6, label="50% of runs")

ax.legend(handles=[line_left_proxy, line_right, band95_proxy, band50_proxy],
          loc="upper right")

plt.tight_layout()
out = Path(__file__).parent / "ma_shortrep.png"
fig.savefig(out, dpi=150, bbox_inches="tight")

# %%
# sweeping across whole range of repression

# %%
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ABM_PATH = Path(__file__).parent / "resilience_ABM.py"
 
spec = importlib.util.spec_from_file_location("resilience_ABM", ABM_PATH)
abm = importlib.util.module_from_spec(spec)
sys.modules["resilience_ABM"] = abm
spec.loader.exec_module(abm)
 
adj_matrix = abm.adj_matrix
nodes_df = abm.nodes_df
ModelConfig = abm.ModelConfig
build_graph = abm.build_graph
run_all = abm.run_all
timestep_update = abm.timestep_update
log_state = abm.log_state
empty_history = abm.empty_history
 
cfg = ModelConfig()
 
 
def sat(x):
    return 1 - x**2


def final_resilience(run_arr: np.ndarray) -> float:
    """Return the last non-nan value in a single run's time series.

    Arrays from run_all are nan-padded past the point a run ends
    (e.g. 'group dissolved'), so this grabs the last real value reached.
    """
    valid = run_arr[~np.isnan(run_arr)]
    return valid[-1] if len(valid) > 0 else np.nan


def summarise(final_vals: np.ndarray):
    """mean, SEM, and count of non-nan entries in an array of per-run final values."""
    final_vals = final_vals[~np.isnan(final_vals)]
    n = len(final_vals)
    if n == 0:
        return np.nan, np.nan, 0
    mean = np.mean(final_vals)
    sem = np.std(final_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, sem, n


def sweep_repression(adj_matrix, nodes_df, cfg, t_repend_values,
                      metrics=("group_resilience", "mean_individual_resilience")):
    """Run the model once per t_repend value and collect the final value of each
    requested metric from every run, then summarise as mean +/- standard error.

    Returns t_used plus a dict keyed by metric name, each holding (mean, sem, n_valid)
    arrays over t_repend_values.
    """
    out = {m: {"mean": [], "sem": [], "n_valid": []} for m in metrics}

    for t_repend in t_repend_values:
        cfg.t_repend = int(t_repend)
        results, _, _ = run_all(adj_matrix, nodes_df, cfg)

        for m in metrics:
            final_vals = np.array([final_resilience(run_arr) for run_arr in results[m]])
            mean, sem, n = summarise(final_vals)
            out[m]["mean"].append(mean)
            out[m]["sem"].append(sem)
            out[m]["n_valid"].append(n)

    t_used = np.asarray(t_repend_values, dtype=float)
    for m in metrics:
        out[m] = {k: np.asarray(v) for k, v in out[m].items()}
    return t_used, out


# %%
# run the sweep: t_repend from 14 to 300 in steps of 5
t_repend_values = np.arange(14, 300 + 1, 5)
t_used, sweep_out = sweep_repression(
    adj_matrix, nodes_df, cfg, t_repend_values,
    metrics=("group_resilience", "mean_individual_resilience"),
)

# %%
import pandas as pd
 
csv_path = Path(__file__).parent / "ma_sweep.csv"
 
rows = []
for m in sweep_out:
    for t, mean, sem, n in zip(t_used, sweep_out[m]["mean"], sweep_out[m]["sem"], sweep_out[m]["n_valid"]):
        rows.append({
            "t_repend": t,
            "metric": m,
            "mean": mean,
            "sem": sem,
            "n_valid": n,
        })
sweep_df = pd.DataFrame(rows)
sweep_df.to_csv(csv_path, index=False)

# %%
# plot: mean final resilience per repression value, with SEM error bars
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

axes[0].errorbar(
    t_used, sweep_out["group_resilience"]["mean"], yerr=sweep_out["group_resilience"]["sem"],
    fmt="o-", color="darkred", ecolor="gray",
    elinewidth=1, capsize=3, markersize=5,
)
axes[0].set_ylabel(r"Final group resilience $R$")
axes[0].set_title("Final group resilience across repression sweep (mean ± SEM)")
axes[0].grid(alpha=0.3)

axes[1].errorbar(
    t_used, sweep_out["mean_individual_resilience"]["mean"],
    yerr=sweep_out["mean_individual_resilience"]["sem"],
    fmt="o-", color="steelblue", ecolor="gray",
    elinewidth=1, capsize=3, markersize=5,
)
axes[1].set_xlabel("t_repend (time of repression decrease)")
axes[1].set_ylabel(r"Final mean individual resilience $\hat{r}$")
axes[1].set_title("Final mean individual resilience across repression sweep (mean ± SEM)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
# %%

# ------- PLOT: empirical_sweep.csv vs ma_sweep.csv (APA style) ---------
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).parent

# %%
# load both sweep result sets and tag their source
empirical_df = pd.read_csv(Path(__file__).parent / "empirical_sweep.csv")
ma_df = pd.read_csv(Path(__file__).parent / "ma_sweep.csv")

empirical_df["source"] = "Ego Projection"
ma_df["source"] = "Memory-Based Strategy"

combined_df = pd.concat([empirical_df, ma_df], ignore_index=True)

# %%
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

STYLE_BY_SOURCE = {
    "Ego Projection": {"color": "black", "marker": "o", "linestyle": "-"},
    "Memory-Based Strategy": {"color": "0.5", "marker": "s", "linestyle": "--"},
}


def plot_metric(ax, df, metric, ylabel):
    for source, sub in df[df["metric"] == metric].groupby("source"):
        sub = sub.sort_values("t_repend")
        style = STYLE_BY_SOURCE.get(source, {})
        ax.errorbar(
            sub["t_repend"], sub["mean"], yerr=sub["sem"],
            capsize=3, markersize=5, linewidth=1.2, label=source,
            **style,
        )
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


# %%
# Figure 1: individual and group resilience, both data sources overlaid
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

plot_metric(axes[0], combined_df, "mean_individual_resilience",
            "Mean Individual Resilience")
plot_metric(axes[1], combined_df, "group_resilience",
            "Mean Group Resilience")

axes[1].set_xlabel("Time of Repression Decrease")

fig.suptitle("Individual and Group Resilience Across Repression Durations",
             fontsize=13, y=0.98)


plt.tight_layout()
#plt.savefig(Path(__file__).parent / "resilience_comparison_apa.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
print("insividual resilience")
d1 = stats["mean_individual_resilience"]   
mean, p025, p975 = (np.asarray(d1[k], float) for k in ("mean", "p025", "p975"))

summary = {
    "max of mean": (np.nanmax(mean),  np.nanargmax(mean)),
    "min of mean": (np.nanmin(mean), np.nanargmin(mean)),
    "min of p025": (np.nanmin(p025),  np.nanargmin(p025)),
    "max of p025": (np.nanmax(p025),  np.nanargmax(p025)),
    "min of p975": (np.nanmin(p975),  np.nanargmin(p975)),
    "max of p975": (np.nanmax(p975),  np.nanargmax(p975)),
}

for name, (value, t) in summary.items():
    print(f"{name}: {value:.3f} at t = {t}")

print("group resilience")
d2 = stats["group_resilience"]   
mean, p025, p975 = (np.asarray(d2[k], float) for k in ("mean", "p025", "p975"))

summary = {
    "max of mean": (np.nanmax(mean),  np.nanargmax(mean)),
    "min of mean": (np.nanmin(mean), np.nanargmin(mean)),
    "min of p025": (np.nanmin(p025),  np.nanargmin(p025)),
    "max of p025": (np.nanmax(p025),  np.nanargmax(p025)),
    "min of p975": (np.nanmin(p975),  np.nanargmin(p975)),
    "max of p975": (np.nanmax(p975),  np.nanargmax(p975)),
}

for name, (value, t) in summary.items():
    print(f"{name}: {value:.3f} at t = {t}")

print("causes of burnout")
d3 = stats["causes_of_burnout"]
mean = np.asarray(d3["mean"], float)

summary = {
    "max of mean": (np.nanmax(mean), np.nanargmax(mean)),
    "min of mean": (np.nanmin(mean), np.nanargmin(mean)),
}

for name, (value, t) in summary.items():
    print(f"{name}: {value:.3f} at t = {t}")

print("average internal social support")
d4 = stats["internal_social_support"]
mean = np.asarray(d4["mean"], float)

summary = {
    "max of mean": (np.nanmax(mean), np.nanargmax(mean)),
    "min of mean": (np.nanmin(mean), np.nanargmin(mean)),
}

for name, (value, t) in summary.items():
    print(f"{name}: {value:.3f} at t = {t}")
# %%
metrics = ["mean_individual_resilience", "group_resilience"]

summary = (
    combined_df[combined_df["metric"].isin(metrics)]
    .pivot_table(index="t_repend", columns=["metric", "source"], values="mean")
    .round(3)
)
print(summary.to_string())
# %%
