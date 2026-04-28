# PREPARATIONS
# %%
import importlib.util
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "group_resilience":  "Initial SMO Resilience",
    "causes_of_burnout": "Initial Causes of Burnout",
    "rep_low":           "Repression Low (rep_low)",
    "rep_high":          "Repression High (rep_high)",
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
    #out = Path(__file__).parent / f"ofat_{rep_param}_timeseries.png"
    #fig.savefig(out, dpi=150, bbox_inches="tight")
    #plt.close(fig)

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
#out = Path(__file__).parent / "ofat_displacement_summary.png"
#fig.savefig(out, dpi=150, bbox_inches="tight")
#plt.close(fig)
# %%

