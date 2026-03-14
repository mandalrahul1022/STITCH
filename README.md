# University Malware SEIR Simulation

Agent-based simulation of a malware outbreak on a university network. The model uses
SEIR dynamics (Susceptible, Exposed, Infected, Recovered) on a scale-free network to
show how hubs and device types influence spread and how different patching strategies
change outcomes.

## Highlights
- Network-driven contagion (Barabasi-Albert scale-free graph)
- Heterogeneous devices (Servers, Lab PCs, Student Laptops) with different security levels
- Incubation delay (E -> I) and stochastic transmission
- Intervention policies: random vs targeted hub-first patching
- Interactive Mesa dashboard + time-series charts

## Requirements
- Python 3.10+ (tested with Python 3.11)
- Packages listed in `requirements.txt`

## Setup
```bash
pip install -r requirements.txt
```

## Run the interactive simulation
```bash
python3 server.py
```
Then open: http://127.0.0.1:8521

## Run a headless scenario (no UI)
```bash
python3 - <<'PY'
from models.university_network import UniversityNetwork, HealthState

model = UniversityNetwork(num_agents=200, initial_outbreak_size=5, seed=42)
for _ in range(50):
    model.step()

print("Final infected:", model.count_state(HealthState.INFECTED))
PY
```

## Project structure
- `models/university_network.py`: Core SEIR agent model and network logic
- `server.py`: Mesa visualization server with charts and controls
- `requirements.txt`: Python dependencies
- `phase2.ipynb`, `run_analysis.ipynb`: Analysis notebooks

## Key parameters (UI sliders)
- `num_agents`: total number of computers in the network
- `avg_incubation_time`: average days to progress from Exposed to Infected
- `virus_spread_chance`: base transmission probability
- `patching_rate`: fraction of infected machines patched per day
- `initial_outbreak_size`: number of initially infected machines

## Notes
- The network uses preferential attachment to model hubs typical of real systems.
- Targeted patching is expected to outperform random patching in hub-dominated networks.

## Mathematical Architecture & Generalizability
The engine is substrate-agnostic: it computes stochastic SEIR-style compartment transitions over a Barabasi-Albert scale-free topology, then applies intervention as a controllable process (random or degree-targeted patching). Because spread is simulated on an explicit graph with heterogeneous node security, the same framework can be retuned to other host/pathogen-like contagion domains where network structure and intervention policy jointly determine cascade behavior.

## Key Results
The notebook `run_analysis.ipynb` now reports reproducible Monte Carlo metrics using fixed seeds (`seed=trial_index`) with `N=200` and initial outbreak size `5`.

1. **Percolation sweep (transmission space, no intervention):**
   - Across `virus_spread_chance in [0.02, 1.00]`, outbreak probability (`CumulativeExposed > 100`) is saturated at `1.0`.
   - This implies the critical transmission threshold is below the tested range: **critical < 0.020** for BA(200, m=3) under zero patching.

2. **Patching-rate sensitivity (Random strategy, spread=0.4):**
   - Mean peak infected at `patching_rate=0.05`: **126.94**
   - Mean peak infected at `patching_rate=0.10`: **85.18**
   - Doubling patching rate reduces mean peak infection by **32.90%**

3. **Strategy comparison (spread=0.4, patching_rate=0.10):**
   - Mean peak infected with `Random`: **85.18**
   - Mean peak infected with `Targeted`: **40.68**
   - Targeted hub patching reduces mean peak infection by **52.24%** versus random at the same rate.

