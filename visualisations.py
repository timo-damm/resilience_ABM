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
 
cfg = ModelConfig()
 
 
def sat(x):
    return 1 - x**2



# ---------------------------------------------------------------------
# ODE system, now forced by the real (time-varying) repression schedule
# instead of a fixed rho
# ---------------------------------------------------------------------
def rhs(t, y, cfg):
    rhat, R, B = y
    rho = repression_schedule(t, cfg)

    rhat_term = max(0.39 - rhat, 0.0)  # lower bound of iss (a norm)

    drhat = sat(rhat) * 0.015 * (-rho + 0.6 + rhat_term + R - B)
    dR = sat(R) * 0.015 * (-rho + 0.39)
    dB = sat(B) * 0.015 * (-rhat - R)

    if B <= 0 and dB < 0:
        dB = 0.0

    return [drhat, dR, dB]


# ---------------------------------------------------------------------
# single real ABM run, mirroring run_all() but for one realization only
# ---------------------------------------------------------------------
def run_single_abm(adj_matrix, nodes_df, cfg):
    G = build_graph(adj_matrix, nodes_df, cfg)
    history = empty_history()
    for t in range(cfg.T):
        G.graph["repression"] = repression_schedule(t, cfg)
        result = timestep_update(G, cfg)
        if result == "group dissolved":
            break
        support_received, support_given, n_dropout, n_joined = result
        log_state(G, t, history, support_received, support_given)
    return history


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
norm = plt.Normalize(0, 0.39 + 0.61)

for y0 in initial_conditions:
    sol = solve_ivp(
        rhs, tspan, y0, args=(cfg,), t_eval=t_eval, rtol=1e-8, atol=1e-8
    )

    rhat = sol.y[0]
    B = sol.y[2]

    baseline = np.clip(0.39 - rhat, 0.3, 1)
    suppress_start, suppress_end = -0.4, -0.8
    depth = np.clip((suppress_start - rhat) / (suppress_start - suppress_end), 0, 1)
    suppression = (1 - depth) ** 2
    iss_hat = baseline * suppression

    points = np.array([B, rhat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(iss_hat[:-1])
    ax.add_collection(lc)

    step = 10
    min_move = 0.02
    for i in range(step, len(B) - 1, step):
        dB = B[i + 1] - B[i]
        dr = rhat[i + 1] - rhat[i]
        if np.hypot(dB, dr) < min_move:
            continue
        ax.annotate(
            "",
            xy=(B[i] + dB, rhat[i] + dr),
            xytext=(B[i], rhat[i]),
            arrowprops={
                "arrowstyle": "->",
                "color": "k",
                "lw": 1,
                "mutation_scale": 12,
                "alpha": 0.8,
            },
        )

    ax.plot(B[0], rhat[0], "o", color="k", markersize=2, alpha=0.6)

ax.add_patch(
    Rectangle(
        (0, -1), 1, 2,
        facecolor="white", alpha=0.5, edgecolor="none", zorder=5,
    )
)

# ---------------------------------------------------------------------
# highlighted trajectory: a REAL ABM run, not an ODE integration
# ----------------------------------------------------------
history = run_single_abm(adj_matrix, nodes_df, cfg)
 
t_hist = np.array(history["t"])
rhat_emp = np.array(history["mean_individual_resilience"])
B_emp = np.array(history["causes_of_burnout"])
iss_emp = np.array(history["internal_social_support"])
R_emp = np.array(history["group_resilience"])

 

points = np.array([B_emp, rhat_emp]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(
    segments, cmap=cmap, norm=norm, linewidth=4, alpha=1.0, zorder=10
)
lc.set_array(iss_emp[:-1])
ax.add_collection(lc)
ax.plot(B_emp[0], rhat_emp[0], "ko", markersize=6, zorder=11)
ax.axhline(0.0, linestyle="--", color="grey")
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("Causes of Burnout", fontsize=14)
ax.set_ylabel("Resilience", fontsize=14)
ax.set_title("Empirical Variant, 55 Weeks of High Repression", fontsize=14)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label("Average Internal Social Support", fontsize=14)

ax_right = ax.twinx()
ax_right.set_ylim(-1, 1)
line_right, = ax_right.plot(
    B_emp, R_emp,
    linestyle=":", color="black", linewidth=2,
    label="Average SMO resilience"
)
ax_right.set_yticks([])
ax_right.tick_params(right=False, labelright=False)

cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

line_left_proxy = Line2D(
    [0], [0], color=cmap_obj(0.5), linewidth=4,
    label="Average individual resilience "
)

ax.legend(handles=[line_left_proxy, line_right], loc="upper right")

plt.tight_layout()
out = Path(__file__).parent / "empirical_shortrep.png"
#fig.savefig(out, dpi=150, bbox_inches="tight")

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
 
csv_path = Path(__file__).parent / "empirical_sweep.csv"
 
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

empirical_df["source"] = "Empirical Variant"
ma_df["source"] = "Mutual Aid Variant"

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
    "Empirical Variant": {"color": "black", "marker": "o", "linestyle": "-"},
    "Mutual Aid Variant": {"color": "0.5", "marker": "s", "linestyle": "--"},
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

fig.suptitle("Individual and Group Resilience Across Repression Sweep",
             fontsize=13, y=0.98)


plt.tight_layout()
plt.savefig(Path(__file__).parent / "resilience_comparison_apa.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
