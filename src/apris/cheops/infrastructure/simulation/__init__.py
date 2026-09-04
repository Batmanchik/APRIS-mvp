"""Event-level simulation of financial flows.

Produces raw ``TransactionEvent`` streams only. No engineered features are
written by the generator: every feature must be derived by the detection
layers from the events themselves. See docs/SIMULATION_LAYER.md.
"""

from apris.cheops.infrastructure.simulation.config import (
    EvasionKnobs,
    SimulationConfig,
)
from apris.cheops.infrastructure.simulation.generator import (
    SimulatedWorld,
    generate_world,
)

__all__ = [
    "EvasionKnobs",
    "SimulationConfig",
    "SimulatedWorld",
    "generate_world",
]
