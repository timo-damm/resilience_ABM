A multi-level Agent Based Model (ABM) of resilience at the individual and group level. Tailored to social movement organisations (SMOs), and based on extended fieldwork with an SMO. Find more 
details about the fieldwork [here](https://osf.io/9jbyr/overview?view_only=66495dfcfc2d4f00a2c9c20b89fe1fb6)

more descriptions to follow

## model overview:
_environment_:
- external stressors (affects SMO, level and changes vary by scenario)
- external social support (affects individuals, distribution in `synthetic_nodes.csv`)

_variables_: 
- SMO resilience (network-level variable, initialised at 0) 
- individual resilience (individual-level variable, initialised with normal distribution) 
- internal stressors (network-level variable, initialised at 0.2)


_mechanisms_:
- agents dropping out: threshold 
- agents joining: rate proportional to repression, based on parameterisation, social support drawn from current distribution of the group
- edge updating: newcomers joining  make 2 connections, for everyone else, probabilistic edge updating with anyone in group, probability influenced by individual level of burnout, also include edge deletion with low probability
- internal social support: mechanism manipulated. Generally determined based on perceived need of neighbours, neighbours' resilience and resilience dependent capacity to give social support

- external stressors negatively influences individual and SMO resilience
- external social support positively influences individual resilience
- internal stressors have negative feedback loop with individual resilience and are negatively influenced by SMO resilience 
- internal social support has positive feedback with individual and SMO resilience
- individual and SMO resilience have positive feedback loop with each other


## Setup & Installation

_1. Install uv_

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or on macOS with Homebrew:

```bash
brew install uv
```

_2. Clone the repository_

```bash
git clone git@github.com:Liz490/political_power_abm.git
cd political_power_abm
```

_3. Create a virtual environment and install dependencies_

```bash
uv sync
```
This will create a `.venv` directory and install all dependencies pinned in `uv.lock`.

_4. Activate the virtual environment_

```bash
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows
```

## Using the Model
Everything should work once dependencies are installed.
To switch between mechanisms for internal social support, change `n` to `nb` in the `timestep_update()` function in `resilience_abm.py` (line 167):

_ego-projection strategy_:
```needs = {
            nb: max(0.0, cfg.tau - G.nodes[n]["social_support"] - G.nodes[n]["individual_resilience"]) #change between nb and n here to adjust between the two variants
            for nb in neighbors
        }
```

_memory-based strategy_:
```needs = {
            nb: max(0.0, cfg.tau - G.nodes[nb]["social_support"] - G.nodes[n]["individual_resilience"]) #change between nb and n here to adjust between the two variants
            for nb in neighbors
        }
```


