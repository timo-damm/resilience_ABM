# ------- PREPARATION ---------
# %%
# preparation and setup
import networkx as nx                         
import numpy as np                             
import matplotlib.pyplot as plt 
import pandas as pd
import random
from pathlib import Path
from dataclasses import dataclass

# import data
BASE_DIR = Path(__file__).parent
adj_matrix = pd.read_csv(BASE_DIR / "synthetic_edges.csv", index_col=0)
adj_matrix.columns = adj_matrix.columns.astype(int)
nodes_df = pd.read_csv(BASE_DIR / "synthetic_nodes.csv")

# %%
# model configuration
@dataclass
class ModelConfig:
    # social support
    tau: float = 1.0 #total need for support
    contribution_max = 1.0 #group norm for how much support is adequate at max (minimum defined in contributions function)
    external_support_weight: float = 0.01 # external social support effect weight (on resilience)
    internal_support_weight: float = 0.01 # internal social support effect weight (on resilience)

    #resilience
    #initiation function for resilience distribution here
    micro_meso_weight: float = 0.01 # micro resilience on meso resilience effect weight
    meso_micro_weight: float = 0.01 # meso resilience on micro resilience effect weight
    resilience_cob_weight: float = -0.01
    resilience_iss_weight: float = 0.01 #

    #causes of burnout
    causes_burnout_weight: float = -0.01 # causes of burnout effect weight on resilience

    #repression (think stresors in model)
    repression_weight = -0.01 #weight of repression on everything

    #network dynamics
    dropout_threshold: float = -0.8 # individual resilience threshold for dropping out
    support_threshold: float = -0.5 # individual resilience threshold for supporting others
    edge_base_prob: float = 0.1 #base probability of forming an edge
    edge_resilience_weight: float = 0.06 # resilience effect weight on edge formation probability
    base_rate: float = 0.217 # base rate of agents joining
    new_agent_connections: int = 2 # number of new connections by new agents
    social_support_bias = 0.1 #slight bias for higher social support than currently existing

    #repression schedule/intensity
    rep_low: float = 0.2 # minimum value of repression
    rep_high: float = 0.8 # maximum value of repression
    t_low: int = 40 # time of low repression
    t_transition: int = 5 # time of repression increase
    t_repend: int = 150 # time of repression decresase
    repression_weight = -0.1 #weight of repression on everything

    # simulation
    T: int = 200
    num_runs: int = 100

cfg = ModelConfig()

# %%
# helper functions
def build_graph(adj_matrix: pd.DataFrame,
                nodes_df: pd.DataFrame,
                cfg: ModelConfig) -> nx.Graph:
    G = nx.from_pandas_adjacency(adj_matrix)

    soc_sup = dict(zip(nodes_df["ID"], nodes_df["SOC_SUP"] / 10.0))
    nx.set_node_attributes(G, soc_sup, "social_support")

    for n in G.nodes():
        G.nodes[n]["individual_resilience"] = np.clip(
            np.random.normal(0.0, 0.3), -1, 1
        )

    G.graph.update({
        "internal_social_support": 0.0, #basically just a placeholder bc individual contributions are computed in first step
        "group_resilience": 0.0,
        "causes_of_burnout": 0.2,
        "repression": cfg.rep_low,
    })
    return G # builds graph from synthetic data

def sat(r: float) -> float:
    return 1 - r ** 2 # saturation function for variables near bounds

def repression_schedule(t: int, cfg: ModelConfig) -> float:
    lo, hi = cfg.rep_low, cfg.rep_high
    if t < cfg.t_low:
        return lo
    elif t < cfg.t_low + cfg.t_transition:
        return lo + (hi - lo) * (t - cfg.t_low) / cfg.t_transition
    elif t < cfg.t_repend:
        return hi
    elif t < cfg.t_repend + cfg.t_transition:
        return hi - (hi - lo) * (t - cfg.t_repend) / cfg.t_transition
    return lo # repression scheduling


# %% 
# initial checks (preliminary visualisation)
G = build_graph(adj_matrix, nodes_df, cfg)
soc_sup = nx.get_node_attributes(G, "social_support")
node_colors = [soc_sup[node] for node in G.nodes()]
pos = nx.spring_layout(G, seed=42)

nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    cmap=plt.cm.viridis,   # heatmap-style colormap
    node_size=300
)

nx.draw_networkx_edges(G, pos, alpha=0.4) # preliminary network visualisation

# %%
# intial checks (network statistics)
degrees = dict(G.degree())
deg_values = np.array(list(degrees.values()))

print(f"Average degree: {deg_values.mean()}")
print(f"Min degree: {deg_values.min()}")
print(f"Max degree: {deg_values.max()}")
print(f"density: {nx.density(G)}")
print(f"average clustering: {nx.average_clustering(G)}")

plt.hist(deg_values, bins=20) # degree distribution

# --------- SIMULATIONS ---------
# %%
# defining main loop
def timestep_update(G, cfg: ModelConfig):
    nodes = list(G.nodes())

    internal_support = G.graph["internal_social_support"]
    micro_mean = np.mean([G.nodes[n]["individual_resilience"] for n in nodes])
    mean_ext_sup = np.mean([G.nodes[n]["social_support"] for n in nodes])
    causes_burnout = G.graph["causes_of_burnout"]
    g_old = G.graph["group_resilience"]
    new_resilience = {}
    support_received = {n: 0.0 for n in nodes}
    support_given = {n: 0.0 for n in nodes}
    support_givers = {n: 0 for n in nodes}  # count how many neighbours gave to each node

    for n in nodes:
        r_n = G.nodes[n]["individual_resilience"]
        neighbors = list(G.neighbors(n))
        if not neighbors or r_n < cfg.support_threshold:
            continue

        need_contribution = max(0.0, min(cfg.contribution_max,
            cfg.tau - G.nodes[n]["social_support"]
        ))

        for nb in neighbors:
            r_nb = G.nodes[nb]["individual_resilience"]
            support_to_nb = max(0.0,
                need_contribution - r_nb + cfg.repression_weight * G.graph["repression"]
            )
            support_received[nb] += support_to_nb
            support_given[n] += support_to_nb
            support_givers[nb] += 1

    # normalise by number of givers so degree doesn't inflate received support
    for n in nodes:
        if support_givers[n] > 0:
            support_received[n] /= support_givers[n]

    mean_iss = np.mean(list(support_received.values()))
    G.graph["internal_social_support"] = mean_iss


    # causes of burnout
    G.graph["causes_of_burnout"] = max(0.0,
        G.graph["causes_of_burnout"] + sat(G.graph["causes_of_burnout"]) * (
            cfg.resilience_cob_weight * (micro_mean + G.graph["group_resilience"])
        )
    )

    for n in nodes:
        external_support = G.nodes[n]["social_support"]
        r_old = G.nodes[n]["individual_resilience"]

        r_new = r_old + sat(r_old) * (
            cfg.repression_weight                * G.graph["repression"]
            + cfg.external_support_weight        * external_support
            + cfg.internal_support_weight        * support_received[n]
            + cfg.meso_micro_weight              * g_old
            + cfg.causes_burnout_weight          * causes_burnout
        )
        new_resilience[n] = r_new

    for n, r in new_resilience.items():
        G.nodes[n]["individual_resilience"] = r

    dropouts = [n for n in nodes if G.nodes[n]["individual_resilience"] < cfg.dropout_threshold]
    G.remove_nodes_from(dropouts)

    nodes = list(G.nodes())

    # stop simulation if everybody dropped out (to not get errors when trying to compute values later)
    if len(nodes) == 0:
        return "group dissolved"

    # group resilience 
    g_new = g_old + sat(g_old) * (
        cfg.repression_weight                * G.graph["repression"]
        + cfg.internal_support_weight        * internal_support
        + cfg.micro_meso_weight              * micro_mean
        + cfg.causes_burnout_weight          * causes_burnout
        + cfg.external_support_weight        * mean_ext_sup
    )
    G.graph["group_resilience"] = g_new

    # new agents joining
    if random.random() < cfg.base_rate * G.graph["repression"]:
        new_id = max(G.nodes()) + 1 if G.nodes() else 1
        existing_ss = [G.nodes[n]["social_support"] for n in G.nodes()]
        G.add_node(
            new_id,
            social_support= cfg.social_support_bias + np.random.choice(existing_ss),
            individual_resilience=np.clip(np.random.normal(0, 0.3), -1, 1)
        )
        targets = random.sample(
            [n for n in G.nodes() if n != new_id],
            min(cfg.new_agent_connections, G.number_of_nodes() - 1)
        )
        G.add_edges_from((new_id, t) for t in targets)

    # edge updating
    for n in list(G.nodes()):
        r_n = G.nodes[n]["individual_resilience"]
        neighbors = set(G.neighbors(n))
        deletion_prob = max(0.0, cfg.support_threshold - r_n)
        addition_prob = max(0.0, cfg.edge_resilience_weight * r_n + cfg.edge_base_prob)

        for neighbor in list(neighbors):
            if random.random() < deletion_prob:
                G.remove_edge(n, neighbor)

        potential_targets = set(G.nodes()) - {n} - set(G.neighbors(n))
        for target in potential_targets:
            if random.random() < addition_prob:
                G.add_edge(n, target)

    return support_received, support_given

#%% 
# logging all variables
def empty_history() -> dict:
    return {
        "t": [],
        "group_resilience": [],
        "mean_individual_resilience": [],
        "std_individual_resilience": [],
        "internal_social_support": [],
        "causes_of_burnout": [],
        "repression": [],
        "num_agents": [],
        "mean_external_social_support": [],
        # cumulative support tracking — one dict per node, accumulated across t
        "cumulative_support_received": {},   # {node_id: total received}
        "cumulative_support_given": {},      # {node_id: total given}
    }

def log_state(G, t, history, support_received, support_given):
    nodes = list(G.nodes())

    if nodes:
        indiv_res = [G.nodes[n]["individual_resilience"] for n in nodes]
        mean_indiv_res = np.mean(indiv_res)
        std_indiv_res = np.std(indiv_res)
        external_support = [G.nodes[n]["social_support"] for n in nodes
                            if "social_support" in G.nodes[n]]
        mean_external_support = np.mean(external_support) if external_support else 0.0
    else:
        mean_indiv_res = std_indiv_res = mean_external_support = 0.0

    history["t"].append(t)
    history["group_resilience"].append(G.graph["group_resilience"])
    history["mean_individual_resilience"].append(mean_indiv_res)
    history["std_individual_resilience"].append(std_indiv_res)
    history["internal_social_support"].append(G.graph["internal_social_support"])
    history["causes_of_burnout"].append(G.graph["causes_of_burnout"])
    history["repression"].append(G.graph["repression"])
    history["num_agents"].append(len(nodes))
    history["mean_external_social_support"].append(mean_external_support)

    # accumulate per-node support across timesteps
    for n, val in support_received.items():
        history["cumulative_support_received"][n] = (
            history["cumulative_support_received"].get(n, 0.0) + val
        )
    for n, val in support_given.items():
        history["cumulative_support_given"][n] = (
            history["cumulative_support_given"].get(n, 0.0) + val
        )

# %%
# running the whole model (for testing with single run, just set num_runs = 1)
def run_all(adj_matrix, nodes_df, cfg):
    keys = ["group_resilience", "mean_individual_resilience", "internal_social_support",
            "causes_of_burnout", "mean_external_social_support", "repression"]
    results = {k: [] for k in keys}
    final_resilience_distributions = []

    for run in range(cfg.num_runs):
        if (run + 1) % 10 == 0:
            print(f"  run {run + 1}/{cfg.num_runs}")
        G = build_graph(adj_matrix, nodes_df, cfg)
        history = empty_history()
        for t in range(cfg.T):
            G.graph["repression"] = repression_schedule(t, cfg)
            result = timestep_update(G, cfg)
            if result == "group dissolved":
                break
            support_received, support_given = result
            log_state(G, t, history, support_received, support_given)
        for k in keys:
            results[k].append(history[k])
        final_resilience_distributions.append(
            np.array([G.nodes[n]["individual_resilience"] for n in G.nodes()])
        )

    return {k: np.array(v) for k, v in results.items()}, final_resilience_distributions

results, final_resilience_distributions = run_all(adj_matrix, nodes_df, cfg)

# %%
# visualisations
plt.figure(figsize=(10, 6))
alpha_val = 0.07

plt.plot(results["group_resilience"].T,            alpha=alpha_val, color="tab:blue")
plt.plot(results["mean_individual_resilience"].T,  alpha=alpha_val, color="tab:orange")
plt.plot(results["causes_of_burnout"].T,           alpha=alpha_val, color="tab:red")
plt.plot(results["internal_social_support"].T,     alpha=alpha_val, color="tab:green")
plt.plot(results["mean_external_social_support"].T,alpha=alpha_val, color="tab:purple")
plt.plot(results["repression"].T,                  alpha=alpha_val, color="black")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Model Volatility Across 100 Runs")
plt.tight_layout()
plt.show()

# %%
# summary table

def summarise_final_step(results: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for var, arr in results.items():
        final = arr[:, -1]  # shape (num_runs,) — last timestep across all runs
        rows.append({
            "variable": var,
            "mean": np.mean(final),
            "min":  np.min(final),
            "max":  np.max(final),
            "std":  np.std(final),
        })
    return pd.DataFrame(rows).set_index("variable").round(4)

summary = summarise_final_step(results)
print(summary)
# %%
# resilience distributions
def plot_individual_resilience_distributions(final_resilience_distributions: list[np.ndarray]):
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, run_values in enumerate(final_resilience_distributions):
        if len(run_values) < 2:
            continue  # skip if too few agents survived
        kde = gaussian_kde(run_values)
        x = np.linspace(-1, 1, 300)
        ax.plot(x, kde(x), color="steelblue", alpha=0.15, linewidth=0.8)

    # overlay the pooled distribution across all runs
    all_values = np.concatenate(final_resilience_distributions)
    kde_pooled = gaussian_kde(all_values)
    x = np.linspace(-1, 1, 300)
    ax.plot(x, kde_pooled(x), color="red", linewidth=2, label=f"pooled (n={len(all_values)})")

    ax.set_xlabel("individual resilience")
    ax.set_ylabel("density")
    ax.set_title("Distribution of individual resilience at t=100 across 100 runs")
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_individual_resilience_distributions(final_resilience_distributions)
# %%
def plot_support_distributions(final_support_received, final_support_given):
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for data, ax, title in [
        (final_support_received, axes[0], "Cumulative support received"),
        (final_support_given,    axes[1], "Cumulative support given"),
    ]:
        all_values = np.concatenate(data)
        x = np.linspace(all_values.min(), all_values.max(), 300)

        for run_values in data:
            if len(run_values) < 2:
                continue
            kde = gaussian_kde(run_values)
            ax.plot(x, kde(x), color="steelblue", alpha=0.15, linewidth=0.8)

        kde_pooled = gaussian_kde(all_values)
        ax.plot(x, kde_pooled(x), color="red", linewidth=2,
                label=f"pooled (n={len(all_values)})")
        ax.set_title(title)
        ax.set_xlabel("cumulative support")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    plt.show()
# %%
