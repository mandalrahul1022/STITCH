"""
Foundation classes for the university SEIR malware simulation.

This module defines the ComputerAgent and UniversityNetwork model that
collectively build the environment (scale-free network) and the smart
agents (computers) with robustness features such as varying security
levels and incubation dynamics.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa.time import RandomActivation


class HealthState(str, Enum):
    """Enumerates the SEIR states available to each computer."""

    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTED = "I"
    RECOVERED = "R"


class ComputerAgent(Agent):
    """
    Represents a single computer on the network with SEIR dynamics.

    Attributes:
        node_type (str): The robustness category ('Server', 'Lab PC', 'Student Laptop').
        security_level (float): Probability of resisting infection (0.0-1.0).
        incubation_timer (int): Days remaining before E -> I transition.
        state (HealthState): Current SEIR compartment.
    """

    def __init__(self, unique_id: int, model: "UniversityNetwork", agent_type: str, security_level: float) -> None:
        super().__init__(unique_id, model)
        self.node_type = agent_type
        self.type = agent_type  # backward compatibility for earlier visualizations
        self.security_level = max(0.0, min(1.0, security_level))
        self.state: HealthState = HealthState.SUSCEPTIBLE
        self.incubation_timer: int = 0

    def step(self) -> None:
        """Advance the agent by one day based on its current state."""
        if self.state == HealthState.INFECTED:
            self._attempt_infections()
        elif self.state == HealthState.EXPOSED:
            self._progress_incubation()

    def expose(self) -> None:
        """Move the agent into the Exposed state and start incubation."""
        if self.state != HealthState.SUSCEPTIBLE:
            return
        self.state = HealthState.EXPOSED
        self.incubation_timer = self.model.draw_incubation_period()
        self.model.daily_new_exposed += 1
        self.model.total_ever_exposed += 1

    def recover(self) -> None:
        """Transition the agent to the Recovered state."""
        self.state = HealthState.RECOVERED
        self.incubation_timer = 0

    def infect(self) -> None:
        """Force the agent into the Infected state, skipping incubation."""
        self.state = HealthState.INFECTED
        self.incubation_timer = 0

    def _attempt_infections(self) -> None:
        """Try to infect susceptible neighbors based on their security levels."""
        for neighbor_id in self.model.graph.neighbors(self.unique_id):
            neighbor_agent = self.model.agent_lookup[neighbor_id]
            if neighbor_agent.state != HealthState.SUSCEPTIBLE:
                continue
            infection_chance = self.model.compute_infection_chance(neighbor_agent)
            if self.random.random() < infection_chance:
                neighbor_agent.expose()

    def _progress_incubation(self) -> None:
        """Count down incubation and switch to Infected when the timer hits zero."""
        self.incubation_timer -= 1
        if self.incubation_timer <= 0:
            self.state = HealthState.INFECTED
            self.incubation_timer = 0
            self.model.daily_new_infected += 1


class UniversityNetwork(Model):
    """
    Mesa model that wires agents onto a Barabási-Albert network.

    Args:
        num_agents (int): Total number of computers in the network.
        initial_outbreak_size (int): Agents that start in the Infected state.
        avg_incubation_time (float): Average days before E -> I transition.
        virus_spread_chance (float): Base infection probability [0, 1].
        patching_rate (float): Fraction of infected machines patched each day.
        patching_strategy (str): 'Random' (default) or 'Targeted' hub-first patching.
        attachment_parameter (Optional[int]): Optional m parameter for BA graph.
        seed (Optional[int]): Optional random seed for reproducibility.
        ensure_server_patient_zero (bool): If True, seed at least one Server infection.
    """

    TYPE_SECURITY = {
        "Server": 0.75,
        "Lab PC": 0.6,
        "Student Laptop": 0.3,
    }

    def __init__(
        self,
        num_agents: int = 1000,
        initial_outbreak_size: int = 10,
        avg_incubation_time: float = 4.0,
        virus_spread_chance: float = 0.4,
        patching_rate: float = 0.1,
        patching_strategy: str = "Random",
        attachment_parameter: Optional[int] = None,
        seed: Optional[int] = None,
        ensure_server_patient_zero: bool = True,
    ) -> None:
        super().__init__()
        if seed is not None:
            self.random.seed(seed)
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")

        self.num_agents = num_agents
        self.initial_outbreak_size = max(0, min(initial_outbreak_size, num_agents))
        self.avg_incubation_time = max(1.0, avg_incubation_time)
        self.virus_spread_chance = max(0.0, min(1.0, virus_spread_chance))
        self.patching_rate = max(0.0, min(1.0, patching_rate))
        self.patching_strategy = patching_strategy
        self.ensure_server_patient_zero = ensure_server_patient_zero
        if self.patching_strategy not in {"Random", "Targeted"}:
            raise ValueError("patching_strategy must be 'Random' or 'Targeted'")

        m = self._resolve_attachment_parameter(attachment_parameter)
        self.graph = nx.barabasi_albert_graph(self.num_agents, m, seed=seed)
        self.grid = NetworkGrid(self.graph)
        self.schedule = RandomActivation(self)
        self.agent_lookup: Dict[int, ComputerAgent] = {}
        self.daily_new_exposed: int = 0
        self.daily_new_infected: int = 0
        self.daily_patched: int = 0
        self.total_ever_exposed: int = 0
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda model: model.count_state(HealthState.SUSCEPTIBLE),
                "Exposed": lambda model: model.count_state(HealthState.EXPOSED),
                "Infected": lambda model: model.count_state(HealthState.INFECTED),
                "Recovered": lambda model: model.count_state(HealthState.RECOVERED),
                "CumulativeExposed": lambda model: model.total_ever_exposed,
                "NewExposed": lambda model: model.daily_new_exposed,
                "NewInfected": lambda model: model.daily_new_infected,
                "Patched": lambda model: model.daily_patched,
                "ServersInfected": lambda model: model.count_by_type(HealthState.INFECTED)["Server"],
                "LabsInfected": lambda model: model.count_by_type(HealthState.INFECTED)["Lab PC"],
                "StudentsInfected": lambda model: model.count_by_type(HealthState.INFECTED)["Student Laptop"],
            }
        )

        self._initialize_agents()
        self._seed_initial_outbreak()
        self.G = self.graph  # compatibility for Mesa's NetworkModule
        self.datacollector.collect(self)

    def step(self) -> None:
        """Advance the simulation by one day."""
        self.daily_new_exposed = 0
        self.daily_new_infected = 0
        self.daily_patched = 0
        self.schedule.step()
        self._apply_patching()
        self.datacollector.collect(self)

    def draw_incubation_period(self) -> int:
        """
        Sample an incubation duration centered around the provided average.

        Returns:
            int: Number of days the agent remains Exposed.
        """
        lower = max(1, int(self.avg_incubation_time * 0.5))
        upper = max(lower, int(self.avg_incubation_time * 1.5))
        return self.random.randint(lower, upper)

    def compute_infection_chance(self, target_agent: ComputerAgent) -> float:
        """
        Blend virus transmissibility with the target's defenses.

        Args:
            target_agent: Neighbor being targeted for infection.

        Returns:
            float: Probability that the neighbor becomes infected.
        """
        adjusted = self.virus_spread_chance * (1.0 - target_agent.security_level)
        return max(0.0, min(1.0, adjusted))

    def count_state(self, state: HealthState) -> int:
        """Count agents currently occupying the specified health state."""
        return sum(1 for agent in self.schedule.agents if agent.state == state)

    def count_by_type(self, state: Optional[HealthState] = None) -> Dict[str, int]:
        """
        Count agents grouped by machine type, optionally filtered by health state.
        """
        totals = {agent_type: 0 for agent_type in self.TYPE_SECURITY}
        for agent in self.schedule.agents:
            if state is None or agent.state == state:
                totals[agent.node_type] += 1
        return totals

    def _initialize_agents(self) -> None:
        type_assignments = self._assign_agent_types()
        for node_id in self.graph.nodes():
            agent_type = type_assignments[node_id]
            security_level = self.TYPE_SECURITY[agent_type]
            agent = ComputerAgent(node_id, self, agent_type, security_level)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node_id)
            self.agent_lookup[node_id] = agent
            self.graph.nodes[node_id].setdefault("agent", []).append(agent)

    def _seed_initial_outbreak(self) -> None:
        if self.initial_outbreak_size == 0:
            return
        patient_zero_ids = self.random.sample(list(self.graph.nodes()), self.initial_outbreak_size)
        if self.ensure_server_patient_zero:
            server_nodes = [node for node, agent in self.agent_lookup.items() if agent.node_type == "Server"]
            if server_nodes:
                if patient_zero_ids:
                    patient_zero_ids[0] = self.random.choice(server_nodes)
                else:
                    patient_zero_ids = [self.random.choice(server_nodes)]
                # drop duplicates while preserving order
                patient_zero_ids = list(dict.fromkeys(patient_zero_ids))
        for node_id in patient_zero_ids:
            self.agent_lookup[node_id].infect()

    def _apply_patching(self) -> None:
        """Simulate the IT department patching infected machines."""
        if self.patching_rate <= 0:
            return

        infected_agents = [agent for agent in self.schedule.agents if agent.state == HealthState.INFECTED]
        if not infected_agents:
            return

        num_to_patch = min(len(infected_agents), int(round(len(infected_agents) * self.patching_rate)))
        if num_to_patch <= 0:
            return

        if self.patching_strategy == "Targeted":
            selected = self.identify_propagation_hubs(infected_agents, num_to_patch)
        else:
            selected = self.random.sample(infected_agents, num_to_patch)

        for agent in selected:
            agent.recover()
            self.daily_patched += 1

    def identify_propagation_hubs(self, infected_agents: list[ComputerAgent], num_to_patch: int) -> list[ComputerAgent]:
        """
        Identify high-impact infected nodes for targeted remediation.

        Identifies structural chokepoints in the network by ranking infected
        nodes by their degree in the Barabasi-Albert topology. High-degree
        nodes act as super-spreaders; prioritizing their remediation disrupts
        the largest number of transmission paths.

        Parameters
        ----------
        infected_agents : list[ComputerAgent]
            Infected agents currently eligible for patching.
        num_to_patch : int
            Number of infected agents to select for patching.

        Returns
        -------
        list[ComputerAgent]
            Top `num_to_patch` infected agents sorted by descending degree.
        """
        ordered = sorted(infected_agents, key=lambda agent: self.graph.degree[agent.unique_id], reverse=True)
        return ordered[:num_to_patch]

    def _assign_agent_types(self) -> Dict[int, str]:
        """Label nodes as Server, Lab PC, or Student Laptop based on degree ranking."""
        sorted_nodes = [node for node, _ in sorted(self.graph.degree, key=lambda item: item[1], reverse=True)]
        server_target = min(5, self.num_agents)
        lab_target = min(max(int(round(self.num_agents * 0.2)), 0), self.num_agents - server_target)

        server_nodes = set(sorted_nodes[:server_target])
        lab_nodes = set(sorted_nodes[server_target : server_target + lab_target])

        assignments: Dict[int, str] = {}
        for node_id in self.graph.nodes():
            if node_id in server_nodes:
                assignments[node_id] = "Server"
            elif node_id in lab_nodes:
                assignments[node_id] = "Lab PC"
            else:
                assignments[node_id] = "Student Laptop"
        return assignments

    def _resolve_attachment_parameter(self, attachment_parameter: Optional[int]) -> int:
        """Derive a valid m parameter for the Barabási-Albert graph."""
        if attachment_parameter is not None:
            m = attachment_parameter
        else:
            # Default to a modest connectivity if not specified.
            m = max(1, min(5, self.num_agents // 20 or 1))

        if m >= self.num_agents:
            m = max(1, self.num_agents - 1)
        return m


def run_scenario(
    num_agents: int = 200,
    initial_outbreak_size: int = 5,
    avg_incubation_time: float = 4.0,
    virus_spread_chance: float = 0.4,
    patching_rate: float = 0.1,
    attachment_parameter: Optional[int] = None,
    steps: int = 100,
) -> pd.DataFrame:
    """
    Run a single simulation and visualize aggregate SEIR trends.

    Returns:
        pd.DataFrame: Time series of Susceptible/Exposed/Infected/Recovered counts.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    model = UniversityNetwork(
        num_agents=num_agents,
        initial_outbreak_size=initial_outbreak_size,
        avg_incubation_time=avg_incubation_time,
        virus_spread_chance=virus_spread_chance,
        patching_rate=patching_rate,
        attachment_parameter=attachment_parameter,
    )

    for _ in range(steps):
        model.step()

    results: pd.DataFrame = model.datacollector.get_model_vars_dataframe()
    if results.empty:
        raise RuntimeError("Simulation produced no data.")

    peak_count = int(results["Infected"].max())
    peak_day = int(results["Infected"].idxmax())

    print(f"Peak Infection Count: {peak_count}")
    print(f"Day of Peak Infection: {peak_day}")

    color_lookup = {
        "Susceptible": "#2ecc71",
        "Exposed": "#f1c40f",
        "Infected": "#e74c3c",
        "Recovered": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for column, color in color_lookup.items():
        ax.plot(results.index, results[column], label=column, color=color, linewidth=2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Computers")
    ax.set_title("University Network SEIR Malware Scenario")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return results
