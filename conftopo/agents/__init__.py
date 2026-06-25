"""ConfTopo-Agent variants for different navigation frameworks."""

from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.agents.etpnav_agent import ConfTopoETPNavAgent
from conftopo.agents.goat_agent_final import (
    ConfTopoGOATAgent,
    ConfTopoGOATAgentFinal,
)
from conftopo.agents.goat_agent_etpnav import (
    ConfTopoGOATAgentETPNav,
    ConfTopoGOATETPNavAgent,
    ETPGoatConfig,
)
from conftopo.agents.goat_agent_etpnav_clean import (
    ConfTopoGOATAgentCleanETPNav,
    ConfTopoGOATCleanETPNavAgent,
    CleanETPGoatConfig,
)

__all__ = [
    "ConfTopoBaseAgent",
    "ConfTopoETPNavAgent",
    "ConfTopoGOATAgent",
    "ConfTopoGOATAgentFinal",
    "ConfTopoGOATAgentETPNav",
    "ConfTopoGOATETPNavAgent",
    "ConfTopoGOATAgentCleanETPNav",
    "ConfTopoGOATCleanETPNavAgent",
    "ETPGoatConfig",
    "CleanETPGoatConfig",
]
