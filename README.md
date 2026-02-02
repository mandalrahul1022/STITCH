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

