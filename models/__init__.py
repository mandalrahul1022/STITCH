"""Package entrypoint for the University SEIR malware simulation models."""

from .university_network import ComputerAgent, HealthState, UniversityNetwork

__all__ = [
    "ComputerAgent",
    "UniversityNetwork",
    "HealthState",
]
