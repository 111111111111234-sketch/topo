"""Base class for ConfTopo-Agent variants."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap
from conftopo.core.instruction_graph import InstructionGraph


class ConfTopoBaseAgent(ABC):
    """Base agent defining the observe → update_memory → plan → act loop."""

    def __init__(self, config: Optional[ConfTopoConfig] = None):
        if config is None:
            config = ConfTopoConfig()
        self.config = config
        self.topo_map = DynamicTopoMap(config.memory)
        self.instruction_graph: Optional[InstructionGraph] = None
        self._step_count = 0

    def reset(self):
        """Reset for new episode (memory cleared)."""
        self.topo_map.reset()
        self.instruction_graph = None
        self._step_count = 0

    def reset_keep_memory(self):
        """Reset for new goal within same episode (memory preserved)."""
        self._step_count = 0

    def set_goal(self, instruction_graph: InstructionGraph):
        self.instruction_graph = instruction_graph

    @abstractmethod
    def observe(self, obs: Dict[str, Any]) -> None:
        """Process current observation and update internal state."""
        ...

    @abstractmethod
    def update_memory(self) -> None:
        """Update DynamicTopoMap with new information from observation."""
        ...

    @abstractmethod
    def plan(self) -> Any:
        """Determine next target based on goal + memory."""
        ...

    @abstractmethod
    def act(self, plan_output: Any) -> Any:
        """Convert plan output to action for the environment."""
        ...

    def step(self, obs: Dict[str, Any]) -> Any:
        """Full agent loop: observe → update → plan → act."""
        self._step_count += 1
        self.topo_map.step()

        self.observe(obs)
        self.update_memory()
        plan_output = self.plan()
        action = self.act(plan_output)
        return action

    @property
    def step_count(self) -> int:
        return self._step_count
