# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def sat(x):
    return 1 - x**2


# ODE system

def rhs(t, y, rho):
    rhat, R, B = y

    drhat = (
        sat(rhat)
        * 0.015
        * (-rho + 0.6 + 0.39 - rhat + R - B)
    )

    dR = (
        sat(R)
        * 0.015
        * (-rho + 0.39)
    )

    dB = (
        sat(B)
        * 0.015
        * (-rhat - R)
    )

    if B <= 0 and dB < 0:
        dB = 0.0

    return [drhat, dR, dB]


# Parameters

tspan = (0, 600)
t = np.linspace(*tspan, 400)

rho = 0.39  # fixed for this plot

#Grid of starting points: one trajectory per (rhat, B) intersection
n_rhat = 20   # number of grid points along rhat
n_B = 20      # number of grid points along B
R_fixed = 0.0  # R held fixed (only matters when rho != 0.39, see earlier discussion)
 
rhat_grid_vals = np.linspace(-1, 1, n_rhat)
B_grid_vals = np.linspace(0, 1, n_B)
rhat_mesh, B_mesh = np.meshgrid(rhat_grid_vals, B_grid_vals, indexing="ij")
 
initial_conditions = np.column_stack([
    rhat_mesh.ravel(),                       # rhat
    np.full(n_rhat * n_B, R_fixed),          # R
    B_mesh.ravel(),                          # B
])


# Plot

fig, ax = plt.subplots(figsize=(9, 7))

cmap = "viridis"
#cmap = LinearSegmentedColormap.from_list("grey_green", ["grey", "green"])
# rhat in [-1, 1] => iss_hat = 0.39 - rhat in [-0.61, 1.39], clipped to >= 0
norm = plt.Normalize(0, 0.39 + 0.61)

for y0 in initial_conditions:
    sol = solve_ivp(
        rhs,
        tspan,
        y0,
        args=(rho,),
        t_eval=t,
        rtol=1e-8,
        atol=1e-8,
    )

    rhat = sol.y[0]
    B = sol.y[2]

    
    # average internal social support, per timestep, clipped at 0
    #iss_hat = np.clip(0.39 - rhat, 0, 1)
 
    baseline = np.clip(0.39 - rhat, 0, 1)
 
    suppress_start, suppress_end = -0.4, -0.8
    depth = np.clip((suppress_start - rhat) / (suppress_start - suppress_end), 0, 1)  # 0 -> 1
    suppression = (1 - depth) ** 2  # 1 at rhat=-0.4, 0 at rhat<=-0.8
 
    iss_hat = baseline * suppression


    # x = B, y = rhat
    points = np.array([B, rhat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=2,
        alpha=0.8,
    )
    lc.set_array(iss_hat[:-1])

    ax.add_collection(lc)

    # Direction arrows every `step` samples, skipping near-stalled points
    # (avoids clumps of overlapping arrows where trajectories slow near
    # the boundaries)
    step = 25
    min_move = 0.02  # minimum (dB, drhat) magnitude to bother drawing
    for i in range(step, len(B) - 1, step):
        dB = B[i + 1] - B[i]
        dr = rhat[i + 1] - rhat[i]

        if np.hypot(dB, dr) < min_move:
            continue

        ax.annotate(
            "",
            xy=(B[i] + dB, rhat[i] + dr),
            xytext=(B[i], rhat[i]),
            arrowprops= {
                "arrowstyle": "->",
                "color": "k",
                "lw": 1,
                "mutation_scale": 12,
                "alpha": 0.8,
            }
        )

    # mark the starting point
    ax.plot(B[0], rhat[0], "o", color="k", markersize=2, alpha=0.6)

ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)

ax.set_xlabel(r"$B$", fontsize=14)
ax.set_ylabel(r"$\hat r$", fontsize=14)
ax.set_title(rf"Phase trajectories at $\rho = {rho}$", fontsize=14)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax, pad=0.02)
cbar.set_label(r"$\widehat{iss}$", fontsize=14)

plt.tight_layout()
out = Path(__file__).parent / "phase_2d_realiss.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
# %%
# different ODE system
