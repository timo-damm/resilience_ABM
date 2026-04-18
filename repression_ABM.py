#!/usr/bin/env python
# coding: utf-8

# ## preparation and initialisation

# %%


# loading libraries 
import networkx as nx                         
import numpy as np                             
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.path import Path as mpPath
from matplotlib.lines import Line2D
import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent

# intitialising graph
adj_matrix = pd.read_csv(BASE_DIR / 'synthetic_edges.csv', index_col= 0 )
#print(adj_matrix)
adj_matrix.columns = adj_matrix.columns.astype(int)
G = nx.from_pandas_adjacency(adj_matrix)

# adding social support
nodes = pd.read_csv(BASE_DIR / 'synthetic_nodes.csv')
soc_sup_dict = dict(zip(nodes['ID'], nodes['SOC_SUP']))
nx.set_node_attributes(G, soc_sup_dict, "social_support")

# rescale to 0-1
for n in G.nodes():
    if "social_support" in G.nodes[n]:
        G.nodes[n]["social_support"] /= 10.0


# ## initial checks to see if everything looks as it should

# %%


# preliminary visualisation
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

nx.draw_networkx_edges(G, pos, alpha=0.4)

# different  -> fix this, person_marker currently not created properly

def person_marker():
    theta = np.linspace(0, 2 * np.pi, 20)
    head = np.column_stack([
        0.0 + 0.22 * np.cos(theta),
        0.35 + 0.22 * np.sin(theta)
    ])

    shoulders = np.array([
        (-0.45, 0.10),
        (-0.30, -0.15),
        (0.00, -0.25),
        (0.30, -0.15),
        (0.45, 0.10),
        (0.30, 0.05),
        (0.00, 0.00),
        (-0.30, 0.05),
        (-0.45, 0.10),
    ])

    verts = np.vstack([head, shoulders])
    codes = (
        [mpPath.MOVETO] +
        [mpPath.LINETO] * (len(head) - 1) +
        [mpPath.MOVETO] +
        [mpPath.CURVE3] * (len(shoulders) - 1)
    )

    return Path(verts, codes)

norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

H = G.copy()
H.remove_nodes_from(list(nx.isolates(H)))

pos = nx.spring_layout(H, seed=42)

norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
cmap = plt.cm.RdYlGn

fig, ax = plt.subplots(figsize=(8, 6))

# draw edges
nx.draw_networkx_edges(H, pos, alpha=0.4, ax=ax)

# prepare data for scatter
xs = [pos[n][0] for n in H.nodes()]
ys = [pos[n][1] for n in H.nodes()]
colors = [soc_sup[n] for n in H.nodes()]

sc = ax.scatter(
    xs,
    ys,
    c=colors,
    cmap=cmap,
    norm=norm,
    s=300,
    marker=person_marker()
)

agent_handle = Line2D(
    [], [],
    marker=person_marker(),
    linestyle="None",
    markersize=14,
    markerfacecolor="green",
    markeredgecolor="green",
    label="agent"
)

ax.legend(handles=[agent_handle], loc="upper right")

# colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("External Social Support")
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

ax.axis("off")
#plt.savefig(BASE_DIR / 'ABM_network.png', dpi = 300)
plt.show()


# %%


# network statistics
degrees = dict(G.degree())
deg_values = np.array(list(degrees.values()))

print(f"Average degree: {deg_values.mean()}")
print(f"Min degree: {deg_values.min()}")
print(f"Max degree: {deg_values.max()}")
print(f"density: {nx.density(G)}")
print(f"average clustering: {nx.average_clustering(G)}")

#degree distribution:
plt.hist(deg_values, bins=20)


# %%


# social support: 
soc_sup = nx.get_node_attributes(G, "social_support")
soc_sup_values = np.array(list(soc_sup.values()))

plt.figure(figsize=(6, 4))
plt.hist(soc_sup_values, bins=20)
plt.show()


# ## Agent Based Model

# %%


# initialise each node as agent (with individual level variables)
for n in G.nodes():    
    G.nodes[n]["individual_resilience"] = np.clip(
        np.random.normal(loc= 0.0, scale=0.3), -1, 1
    ) #mean = 0, stddev = 0.3 scale bound by -1 and 1

    G.nodes[n]["social_support"] = soc_sup[n]  

# define and initialise variables at group level (resilience, internal social support)
G.graph["internal_social_support"] = 0.2
G.graph["group_resilience"] = 0.0
G.graph["causes_of_burnout"] = 0.2
tau = 1.0


# %%


# helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sigmoid_k = 6.0

# repression effect
def repression_effective_weight(
    resilience,
    min_weight,
    max_weight,
    steepness=5.0,
    midpoint=0.0
):

    s = sigmoid(-steepness * (resilience - midpoint))
    return min_weight + (max_weight - min_weight) * s #basically sigmoid function where effect of repression increases with decreasing reslience (more burnout)

repression_weight_min = -0.02
repression_weight_max = -0.02  

#saturation function
def sat(R): 
    sat = 1 - R**2
    return sat    


# %%


# parameters
repression_burnout = 0.0 #0.01          # effect of repression on other causes of burnout (see if I can merge it with other effect of repression)
external_support_weight = 0.01     # external social support effect on individual resilience (maybe also meso)

internal_support_weight = 0.01     # internal social support effect on micro and meso resilience
resilience_iss_weight = 0.01     # resilience effect on internal social support (asymmetric)

causes_burnout_weight = -0.01 #0.02     # causes of burnout weight on resilience
resilience_cob_weight = -0.01      # resilience effect on COB (asymmetric)

micro_meso_weight = 0.01   # individual resilience affecting group resilience
meso_micro_weight = 0.01    # group resilience affecting individual resilience

dropout_threshold = -0.8           # resilience threshold for dropping out
base_rate = 0.217 #0.021631305164                    # base rate for new agents joining
new_agent_connections = 2          # number of connections new agents make
support_threshold = -0.2       # above which edge deletion probability > addition, and agents do not receive support over this edge anymore
edge_base_prob = 0.1          # base probability of forming an edge per timestep
edge_resilience_weight = 0.06  # weight of individual resilience in edge formation


# %%


def repression_schedule(t,
                        low=0.2,
                        high=0.8,
                        t_low=40,
                        t_transition=5,
                        t_repend = 90):
    if t < t_low:
        return low
    elif t < t_low + t_transition:
        return low + (high - low) * (t - t_low) / t_transition
    elif t < t_repend:
        return high
    elif t < t_repend + t_transition:
        return high - (high - low) * (t - t_repend) / t_transition
    else:
        return low


# %%


# main loop
def timestep_update(G):
    nodes = list(G.nodes())

    # update individual resilience
    internal_support = G.graph["internal_social_support"]
    micro_mean = np.mean([G.nodes[n]["individual_resilience"] for n in nodes]) 
    mean_ext_sup = np.mean([G.nodes[n]["social_support"] for n in nodes]) 
    causes_burnout = G.graph["causes_of_burnout"]
    g_old = G.graph["group_resilience"]
    new_resilience = {}

    for n in nodes:
        external_support = G.nodes[n]["social_support"]  # this stays stable
        neighbors = list(G.neighbors(n))

        if neighbors:
            neighbor_resilience = np.mean(
                [G.nodes[j]["individual_resilience"] for j in neighbors]
            ) #computes the mean resilience of the neighbours of n
        else:
            neighbor_resilience = 0.0 #setting it to 0 for isolates

        r_old = G.nodes[n]["individual_resilience"]

        effective_repression_weight = repression_effective_weight(
            r_old,
            repression_weight_min,
            repression_weight_max,
            steepness=sigmoid_k
        )

        r_new = r_old + sat(r_old) *(
            effective_repression_weight * G.graph["repression"]
            + external_support_weight * external_support
            + neighbor_resilience * internal_support_weight * internal_support
            + meso_micro_weight * g_old
            + causes_burnout_weight * causes_burnout
        )

        new_resilience[n] = r_new  # no clipping

    for n, r in new_resilience.items():
        G.nodes[n]["individual_resilience"] = r

    # update group resilience
    g_new = g_old + sat(g_old) *(
        effective_repression_weight * G.graph["repression"]
        + internal_support_weight * internal_support
        + micro_meso_weight * micro_mean
        + causes_burnout_weight * causes_burnout
        + external_support_weight * mean_ext_sup
    )

    G.graph["group_resilience"] = g_new


    #compute individual contributions to social support
    contributions = []

    for n in nodes: 
        external_support = G.nodes[n]["social_support"]
        c_i = max(0.0, tau - external_support)

        contributions.append(c_i)

    G.graph["internal_social_support"] = sum(contributions) / len(contributions)

    #update internal social support with interactions
    G.graph["internal_social_support"] += sat(G.graph["internal_social_support"]) * resilience_iss_weight * G.graph["group_resilience"]


    # update causes of burnout 
    G.graph["causes_of_burnout"] += sat(G.graph["causes_of_burnout"])* (resilience_cob_weight * (micro_mean + G.graph["group_resilience"]) + repression_burnout * G.graph["repression"]) # maybe include direct effect of repression on COB later
    G.graph["causes_of_burnout"] = max(0.0, G.graph["causes_of_burnout"])

    # handle agents dropping out
    dropouts = [n for n in nodes if G.nodes[n]["individual_resilience"] < dropout_threshold]
    G.remove_nodes_from(dropouts)

    # handle agents joining 
    p_join = base_rate * G.graph["repression"]

    if random.random() < p_join:
        new_id = max(G.nodes()) + 1 if len(G.nodes()) > 0 else 1

        # sample social support from existing nodes
        existing_ss = [
            G.nodes[n]["social_support"]
            for n in G.nodes()
            if "social_support" in G.nodes[n]
        ]
        new_soc_sup = 0.1 + np.random.choice(existing_ss) # add small bias for more positive social support

        # add node
        G.add_node(
            new_id,
            social_support=new_soc_sup,
            individual_resilience=np.clip(np.random.normal(0, 0.3), -1, 1)
        )

        # connect to existing nodes
        possible_targets = list(G.nodes())
        possible_targets.remove(new_id)

        if possible_targets:
            targets = random.sample(
                possible_targets,
                min(new_agent_connections, len(possible_targets))
            )
            for t in targets:
                G.add_edge(new_id, t)


    # edge updating for existing agents
    for n in G.nodes():
        neighbors = set(G.neighbors(n))
        potential_targets = set(G.nodes()) - {n} - neighbors
        # Edge deletion probability increases if individual resilience < threshold
        r_n = G.nodes[n]["individual_resilience"]
        deletion_prob = max(0, support_threshold - r_n)  # 0 if above threshold
        addition_prob = max(0, edge_resilience_weight * r_n + edge_base_prob) 
        # Delete edges
        for neighbor in list(neighbors):
            if random.random() < deletion_prob:
                G.remove_edge(n, neighbor)
        # Add edges
        for target in potential_targets:
            if random.random() < addition_prob:
                G.add_edge(n, target)


# %%


# tracking all variables over time
history = {
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

    # individual resilience stats
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

    # store
    history["t"].append(t)
    history["group_resilience"].append(G.graph["group_resilience"])
    history["mean_individual_resilience"].append(mean_indiv_res)
    history["std_individual_resilience"].append(std_indiv_res)
    history["internal_social_support"].append(G.graph["internal_social_support"])
    history["causes_of_burnout"].append(G.graph["causes_of_burnout"])
    history["repression"].append(G.graph["repression"])
    history["num_agents"].append(len(nodes))
    history["mean_external_social_support"].append(mean_external_support)


# %%


# running it for n timesteps: (mainly for checking whether everything works, actual runs should be averaged because of probabilistic nature)
T = 100
for t in range(T):
    G.graph["repression"] = repression_schedule(t)
    timestep_update(G)
    log_state(G, t, history)


# %%


print(G.graph["internal_social_support"])
print(G.graph["group_resilience"])
print(G.graph["causes_of_burnout"])  
#print(G.nodes[n]["individual_resilience"])


# %%


num_runs = 100
T = 100

all_trajectories = []
all_group_res = []
all_mean_indiv_res = []
all_burnout = []
all_internal_support = []
all_external_support = []
all_repression = []

for run in range(num_runs):

    # 🔁 reinitialize the model each run
    G = nx.from_pandas_adjacency(adj_matrix)

    # adding social support
    nodes = pd.read_csv('/home/dot/random_coding/resilience_ABM/synthetic_nodes.csv')
    soc_sup_dict = dict(zip(nodes['ID'], nodes['SOC_SUP']))
    nx.set_node_attributes(G, soc_sup_dict, "social_support")

    # rescale to 0-1
    for n in G.nodes():
        if "social_support" in G.nodes[n]:
            G.nodes[n]["social_support"] /= 10.0
    for n in G.nodes():    
        G.nodes[n]["individual_resilience"] = np.clip(
            np.random.normal(loc= 0.0, scale=0.3), -1, 1
        ) #mean = 0, stddev = 0.3 scale bound by -1 and 1

        G.nodes[n]["social_support"] = soc_sup[n]  

    # define and initialise variables at group level (resilience, internal social support)
    G.graph["internal_social_support"] = 0.2
    G.graph["group_resilience"] = 0.0
    G.graph["causes_of_burnout"] = 0.2
    tau = 1.0

    history = {
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

    for t in range(T):
        G.graph["repression"] = repression_schedule(t)
        timestep_update(G)
        log_state(G, t, history)

    all_group_res.append(history["group_resilience"])
    all_mean_indiv_res.append(history["mean_individual_resilience"])
    all_burnout.append(history["causes_of_burnout"])
    all_internal_support.append(history["internal_social_support"])
    all_external_support.append(history["mean_external_social_support"])
    all_repression.append(history["repression"])


# Convert to arrays
all_group_res = np.array(all_group_res)
all_mean_indiv_res = np.array(all_mean_indiv_res)
all_burnout = np.array(all_burnout)
all_internal_support = np.array(all_internal_support)
all_external_support = np.array(all_external_support)
all_repression = np.array(all_repression)

all_trajectories = np.array(all_trajectories)


# %%


plt.figure(figsize=(10, 6))

alpha_val = 0.07  # same opacity for everything

# Group resilience
plt.plot(all_group_res.T, alpha=alpha_val, color="tab:blue")

# Mean individual resilience
plt.plot(all_mean_indiv_res.T, alpha=alpha_val, color="tab:orange")

# Burnout
plt.plot(all_burnout.T, alpha=alpha_val, color="tab:red")

# Internal social support
plt.plot(all_internal_support.T, alpha=alpha_val, color="tab:green")

# External social support
plt.plot(all_external_support.T, alpha=alpha_val, color="tab:purple")

# repression
plt.plot(all_repression.T, alpha=alpha_val, color="black")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Model Volatility Across 100 Runs")

plt.tight_layout()
plt.show()


# ## visualisations

# %%


plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(6.5, 4.5))  # APA-friendly proportions

ax.plot(history["t"], history["group_resilience"],
        label="Group resilience", linewidth=1.5, color="blue")

ax.plot(history["t"], history["mean_individual_resilience"],
        label="Mean individual resilience", linewidth=1.5, color="orange")

ax.plot(history["t"], history["repression"],
        label="Repression", linewidth=1.2, color="black")

ax.plot(history["t"], history["internal_social_support"],
        label="Internal social support", linewidth=1.5, color="green")

ax.plot(history["t"], history["causes_of_burnout"],
        label="Causes of burnout", linewidth=1.5, color="red")

ax.plot(history["t"], history["mean_external_social_support"],
        label="Mean external social support", linewidth=1.5, color="purple")

ax.axhline(0, linewidth=1, color = "darkgrey")

ax.set_ylim(-1, 1)
ax.set_xlabel("Time step")
ax.set_ylabel("Standardized value")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(True, linewidth=0.5, alpha=0.4)

ax.legend(frameon=False, fontsize=9, loc="best")

plt.tight_layout()
plt.show()


# %%


# visualisation to check outcome
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

nx.draw_networkx_edges(G, pos, alpha=0.4)


# %%


# network statistics
degrees = dict(G.degree())
deg_values = np.array(list(degrees.values()))

print(f"Number of nodes: {nx.number_of_nodes(G)}")
print(f"Average degree: {deg_values.mean()}")
print(f"Min degree: {deg_values.min()}")
print(f"Max degree: {deg_values.max()}")
print(f"density: {nx.density(G)}")
print(f"average clustering: {nx.average_clustering(G)}")

#degree distribution:
plt.hist(deg_values, bins=20)


# %%


print(G.graph["internal_social_support"])
