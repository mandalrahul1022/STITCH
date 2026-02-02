"""
Browser visualization server for the UniversityNetwork SEIR model.
"""

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule, NetworkModule, TextElement

from models import HealthState, UniversityNetwork

STATE_COLORS = {
    HealthState.SUSCEPTIBLE: "green",
    HealthState.EXPOSED: "yellow",
    HealthState.INFECTED: "red",
    HealthState.RECOVERED: "gray",
}


class ProjectDescription(TextElement):
    """Static description displayed on the dashboard."""

    def render(self, model=None) -> str:
        return (
            "<h3>University Malware SEIR Simulation</h3>")


class ChartLabel(TextElement):
    """Small helper to show chart titles/axes explanations in the UI."""

    def __init__(self, title: str, x_axis: str, y_axis: str) -> None:
        super().__init__()
        self.title = title
        self.x_axis = x_axis
        self.y_axis = y_axis

    def render(self, model=None) -> str:  # type: ignore[override]
        return f"<b>{self.title}</b><br>X: {self.x_axis} &nbsp;|&nbsp; Y: {self.y_axis}"


def network_portrayal(G):
    """Map each node to a shape/color sized by robustness tier."""
    portrayal = {"nodes": [], "edges": []}
    for node_id, node_data in G.nodes(data=True):
        agents = node_data.get("agent", [])
        if not agents:
            continue
        agent = agents[0]
        color = STATE_COLORS.get(agent.state, "black")
        shape = "rect" if agent.node_type == "Server" else "circle"
        size = 12 if agent.node_type == "Server" else 6
        portrayal["nodes"].append(
            {
                "id": node_id,
                "shape": shape,
                "size": size,
                "color": color,
                "tooltip": f"{agent.node_type} ({agent.state.value})",
            }
        )

    portrayal["edges"] = [{"source": source, "target": target} for source, target in G.edges()]
    return portrayal


network_module = NetworkModule(network_portrayal, 700, 700)
time_series_label = ChartLabel("S/E/I/R Over Time", "Day/Step", "Number of computers")
chart_module = ChartModule(
    [
        {"Label": "Susceptible", "Color": STATE_COLORS[HealthState.SUSCEPTIBLE]},
        {"Label": "Exposed", "Color": STATE_COLORS[HealthState.EXPOSED]},
        {"Label": "Infected", "Color": STATE_COLORS[HealthState.INFECTED]},
        {"Label": "Recovered", "Color": STATE_COLORS[HealthState.RECOVERED]},
    ]
)

events_label = ChartLabel("Daily Events", "Day/Step", "Event counts (machines)")
events_chart = ChartModule(
    [
        {"Label": "NewExposed", "Color": "#f39c12"},
        {"Label": "NewInfected", "Color": "#c0392b"},
        {"Label": "Patched", "Color": "#7f8c8d"},
    ],
    data_collector_name="datacollector",
)

type_label = ChartLabel("Infected by Device Type", "Day/Step", "Number of infected machines")
type_infected_chart = ChartModule(
    [
        {"Label": "ServersInfected", "Color": "#8e44ad"},
        {"Label": "LabsInfected", "Color": "#2980b9"},
        {"Label": "StudentsInfected", "Color": "#16a085"},
    ],
    data_collector_name="datacollector",
)

model_params = {
    "num_agents": UserSettableParameter("slider", "Number of Agents", 1000, 200, 1500, 50),
    "avg_incubation_time": UserSettableParameter("slider", "Avg Incubation (days)", 4, 1, 14, 1),
    "virus_spread_chance": UserSettableParameter("slider", "Virus Spread Chance", 0.4, 0.1, 1.0, 0.05),
    "patching_rate": UserSettableParameter("slider", "Patching Rate", 0.1, 0.0, 1.0, 0.05),
    "initial_outbreak_size": UserSettableParameter("slider", "Initial Outbreak Size", 10, 1, 50, 1),
}

server = ModularServer(
    UniversityNetwork,
    [
        ProjectDescription(),
        network_module,
        time_series_label,
        chart_module,
        events_label,
        events_chart,
        type_label,
        type_infected_chart,
    ],
    "University Malware SEIR Simulation",
    model_params,
)
server.description = (
    "Malware Simulation: This simulation models a cyber-attack on a university campus to visualize how malware "
    "spreads through a realistic network of student laptops and critical servers. Unlike simple models, it "
    "accounts for hubs which are highly connected computers that act as super-spreaders if infected. This "
    "structure allows you to see the hidden dynamics of a network where not all computers are equal. It "
    "demonstrates why protecting central servers is crucial for preventing a total system collapse.\n\n"
    "You control the outcome by adjusting the virus strength and the IT department response speed. This enables "
    "real-time experiments. The tool specifically lets you compare a standard Random defense against a Targeted "
    "strategy that prioritizes high-risk hubs. This comparison visually proves how smart resource allocation can "
    "flatten the infection curve and minimize network downtime."
)


if __name__ == "__main__":
    server.launch()
