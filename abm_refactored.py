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
    external_support_weight: float = 0.01 # external social support effect weight (on resilience)
    internal_support_weight: float = 0.01 # internal social support effect weight (on resilience)

    #resilience
    micro_meso_weight: float = 0.01 # micro resilience on meso resilience effect weight
    meso_micro_weight: float = 0.01 # meso resilience on micro resilience effect weight
    resilience_cob_weight: float = -0.01
    resilience_iss_weight: float = 0.01 #

    #causes of burnout
    causes_burnout_weight: float = -0.01 # causes of burnout effect weight on resilience

    #repression (think stressors in model)
    repression_weight_min: float = -0.02 # minimum repression weight
    repression_weight_max: float = -0.02 # maximum repression weight
    sigmoid_k: float = 6.0

    #network dynamics
    dropout_threshold: float = -0.8 # individual resilience threshold for dropping out
    support_threshold: float = -0.2 # individual resilience threshold for supporting others
    edge_base_prob: float = 0.1 #base probability of forming an edge
    edge_resilience_weight: float = 0.06 # resilience effect weight on edge formation probability
    base_rate: float = 0.217 # base rate of agents joining
    new_agent_connections: int = 2 # number of new connections by new agents

    #repression schedule/intensity
    rep_low: float = 0.2 # minimum value of repression
    rep_high: float = 0.8 # maximum value of repression
    t_low: int = 40 # time of low repression
    t_transition: int = 5 # time of repression increase
    t_repend: int = 90 # time of repression decresase

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
        "internal_social_support": 0.2,
        "group_resilience": 0.0,
        "causes_of_burnout": 0.2,
        "repression": cfg.rep_low,
    })
    return G # builds graph from synthetic data

def sat(r: float) -> float:
    return 1 - r ** 2 # saturation function for variables near bounds

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x)) # sigmoid  for scaling repression weight

def repression_effective_weight(
    r: float, cfg: ModelConfig
) -> float:
    s = sigmoid(-cfg.sigmoid_k * r)
    return cfg.repression_weight_min + (
        cfg.repression_weight_max - cfg.repression_weight_min
    ) * s # effective repression weight

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

    for n in nodes:
        external_support = G.nodes[n]["social_support"]
        neighbors = list(G.neighbors(n))
        neighbor_resilience = (
            np.mean([G.nodes[j]["individual_resilience"] for j in neighbors])
            if neighbors else 0.0
        )

        r_old = G.nodes[n]["individual_resilience"]
        eff_rep = repression_effective_weight(r_old, cfg)  # ← pass cfg

        r_new = r_old + sat(r_old) * (
            eff_rep                              * G.graph["repression"]
            + cfg.external_support_weight        * external_support
            + cfg.internal_support_weight        * internal_support * neighbor_resilience
            + cfg.meso_micro_weight              * g_old
            + cfg.causes_burnout_weight          * causes_burnout
        )
        new_resilience[n] = r_new

    for n, r in new_resilience.items():
        G.nodes[n]["individual_resilience"] = r

    # group resilience — reuse eff_rep from last node (or compute from g_old)
    eff_rep_group = repression_effective_weight(g_old, cfg)
    g_new = g_old + sat(g_old) * (
        eff_rep_group                        * G.graph["repression"]
        + cfg.internal_support_weight        * internal_support
        + cfg.micro_meso_weight              * micro_mean
        + cfg.causes_burnout_weight          * causes_burnout
        + cfg.external_support_weight        * mean_ext_sup
    )
    G.graph["group_resilience"] = g_new

    # internal social support
    contributions = [
        max(0.0, cfg.tau - G.nodes[n]["social_support"]) for n in nodes
    ]
    G.graph["internal_social_support"] = (
        sum(contributions) / len(contributions)
        + sat(sum(contributions) / len(contributions))
        * cfg.resilience_iss_weight
        * G.graph["group_resilience"]
    )

    # causes of burnout
    G.graph["causes_of_burnout"] = max(0.0,
        G.graph["causes_of_burnout"] + sat(G.graph["causes_of_burnout"]) * (
            cfg.resilience_cob_weight * (micro_mean + G.graph["group_resilience"])
        )
    )

    # dropouts
    dropouts = [n for n in nodes if G.nodes[n]["individual_resilience"] < cfg.dropout_threshold]
    G.remove_nodes_from(dropouts)

    # new agents joining
    if random.random() < cfg.base_rate * G.graph["repression"]:
        new_id = max(G.nodes()) + 1 if G.nodes() else 1
        existing_ss = [G.nodes[n]["social_support"] for n in G.nodes()]
        G.add_node(
            new_id,
            social_support=0.1 + np.random.choice(existing_ss),
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
        "mean_external_social_support": []
    }

def log_state(G, t, history):
    nodes = list(G.nodes())

    if nodes:
        indiv_res = [G.nodes[n]["individual_resilience"] for n in nodes]
        mean_indiv_res = np.mean(indiv_res)
        std_indiv_res = np.std(indiv_res)

        external_support = [
            G.nodes[n]["social_support"]
            for n in nodes
            if "social_support" in G.nodes[n]
        ]
        mean_external_support = np.mean(external_support) if external_support else 0.0
    else:
        mean_indiv_res = 0.0
        std_indiv_res = 0.0
        mean_external_support = 0.0

    history["t"].append(t)
    history["group_resilience"].append(G.graph["group_resilience"])
    history["mean_individual_resilience"].append(mean_indiv_res)
    history["std_individual_resilience"].append(std_indiv_res)
    history["internal_social_support"].append(G.graph["internal_social_support"])
    history["causes_of_burnout"].append(G.graph["causes_of_burnout"])
    history["repression"].append(G.graph["repression"])
    history["num_agents"].append(len(nodes))
    history["mean_external_social_support"].append(mean_external_support)

def run_simulation(adj_matrix: pd.DataFrame, nodes_df: pd.DataFrame,
                   cfg: ModelConfig, T: int = 100) -> dict:
    G = build_graph(adj_matrix, nodes_df, cfg)
    history = empty_history()

    for t in range(T):
        G.graph["repression"] = repression_schedule(t, cfg)
        timestep_update(G, cfg)
        log_state(G, t, history)

    return history


# %%
# running the whole model (for testing with single run, just set num_runs = 1)
def run_all(adj_matrix: pd.DataFrame, nodes_df: pd.DataFrame,
            cfg: ModelConfig, num_runs = 100, T = 100) -> dict[str, np.ndarray]:
    keys = ["group_resilience", "mean_individual_resilience", "internal_social_support",
            "causes_of_burnout", "mean_external_social_support", "repression"]
    results = {k: [] for k in keys}

    for run in range(num_runs):
        if (run + 1) % 10 == 0:
            print(f"  run {run + 1}/{num_runs}")
        history = run_simulation(adj_matrix, nodes_df, cfg, T)
        for k in keys:
            results[k].append(history[k])

    return {k: np.array(v) for k, v in results.items()}

results = run_all(adj_matrix, nodes_df, cfg, num_runs=100, T=100)

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
